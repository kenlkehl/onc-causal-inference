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
import multiprocessing as mp
import os
import queue
import random
import subprocess
import threading
import time
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
    load_single_gpu_store,
    resolve_workers_per_gpu,
    _open_cache_for_worker,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# vLLM server auto-start (multi-GPU)
# ---------------------------------------------------------------------------

def _ensure_vllm_servers(
    server_url: str,
    model_name: str,
    devices: List[str],
    tensor_parallel_size: int = 1,
    download_dir: Optional[str] = None,
    max_model_len: Optional[int] = None,
) -> List[Tuple[Optional[subprocess.Popen], str]]:
    """Start multiple vLLM servers across available GPUs.

    Each server is pinned to specific GPU(s) via CUDA_VISIBLE_DEVICES and
    assigned a unique port. This parallelizes confounder extraction across
    all available GPUs.

    Returns:
        List of (process, url) tuples. Process is None if server was already
        running (only for the base port check).
    """
    import requests
    from urllib.parse import urlparse

    parsed = urlparse(server_url.rstrip('/'))
    base_host = parsed.hostname or 'localhost'
    base_port = parsed.port or 8000

    # Check if a server is already running on the base port
    try:
        resp = requests.get(f"http://{base_host}:{base_port}/health", timeout=5)
        if resp.status_code == 200:
            logger.info(f"vLLM server already running at {server_url}")
            return [(None, server_url)]
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        pass

    # Extract GPU indices from device strings
    gpu_ids = []
    for d in devices:
        if d.startswith("cuda:"):
            gpu_ids.append(int(d.split(":")[1]))
    if not gpu_ids:
        gpu_ids = [0]

    num_servers = max(1, len(gpu_ids) // tensor_parallel_size)
    logger.info(f"Starting {num_servers} vLLM server(s) across {len(gpu_ids)} GPU(s) "
                f"(tensor_parallel_size={tensor_parallel_size})")

    servers = []
    for i in range(num_servers):
        port = base_port + i
        url = f"http://{base_host}:{port}/v1"
        assigned_gpus = gpu_ids[i * tensor_parallel_size : (i + 1) * tensor_parallel_size]
        cuda_visible = ",".join(str(g) for g in assigned_gpus)

        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_name,
            "--port", str(port),
            "--tensor-parallel-size", str(tensor_parallel_size),
            "--gpu-memory-utilization", "0.9",
            "--trust-remote-code",
        ]
        if download_dir:
            cmd.extend(["--download-dir", download_dir])
        if max_model_len:
            cmd.extend(["--max-model-len", str(max_model_len)])

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible

        logger.info(f"Starting vLLM server {i+1}/{num_servers} on port {port} "
                     f"(GPUs: {cuda_visible})")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        servers.append((proc, url))

    # Wait for all servers to become ready
    logger.info("Waiting for vLLM servers to start...")
    time.sleep(30)

    for i, (proc, url) in enumerate(servers):
        port = base_port + i
        health_url = f"http://{base_host}:{port}/health"
        ready = False
        for _ in range(60):  # Up to 5 more minutes each
            try:
                resp = requests.get(health_url, timeout=5)
                if resp.status_code == 200:
                    logger.info(f"vLLM server {i+1}/{num_servers} ready at {url}")
                    ready = True
                    break
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                pass
            time.sleep(5)

        if not ready:
            # Clean up all started servers
            for p, _ in servers:
                if p is not None:
                    p.terminate()
                    p.wait()
            raise RuntimeError(
                f"vLLM server {i+1} on port {port} failed to start within 5 minutes"
            )

    logger.info(f"All {num_servers} vLLM server(s) ready")
    return servers


def _shutdown_vllm_servers(procs: List[subprocess.Popen]):
    """Terminate vLLM server processes and free GPU VRAM."""
    if not procs:
        return
    logger.info(f"Shutting down {len(procs)} vLLM server(s) to free GPU VRAM...")
    for proc in procs:
        proc.terminate()
    for proc in procs:
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            logger.warning(f"vLLM server pid={proc.pid} did not exit, killing...")
            proc.kill()
            proc.wait()
    # Give CUDA time to release memory
    time.sleep(5)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("vLLM servers stopped, GPU VRAM freed")


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
    from oci.models.causal_forest_head import tune_causal_forest_model
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

        def create_causal_forest():
            return CausalForestDML(
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

        cf = create_causal_forest()
        if not tune_causal_forest_model(cf, Y=Y_train, T=T_train, X=X_train):
            cf = create_causal_forest()
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
    flp_chat_template_prompt: Optional[str] = None,
    feature_extractor_type: str = "frozen_llm_pooler",
) -> Dict[str, Any]:
    """Run causal forest with text features + optional confounders.

    Wraps run_causal_forest_experiment with the right config.
    Supports multiple feature extractor types via feature_extractor_type.
    """
    config = ExperimentConfig(
        dataset_path="",  # Not used by the experiment runner directly
        dataset_name="semisynthetic",
        model_type="causal_forest",
        use_explicit_confounders=bool(confounder_specs),
        feature_extractor_type=feature_extractor_type,
        repeat_index=seed - 42,  # For KFold seeding
        flp_model_name=flp_model_name,
        flp_max_length=flp_max_length,
        flp_downprojection_dim=flp_downprojection_dim,
        flp_projection_dim=flp_projection_dim,
        flp_gated_attention_dim=flp_gated_attention_dim,
        flp_chat_template_prompt=flp_chat_template_prompt,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        n_folds=n_folds,
        cf_n_estimators=cf_n_estimators,
        cf_min_samples_leaf=cf_min_samples_leaf,
        gamma_rlearner=gamma_rlearner,
    )

    # For hierarchical_llm, derive hlm_* params from flp_* CLI params
    if feature_extractor_type == "hierarchical_llm":
        config.hlm_model_name = flp_model_name
        config.hlm_downprojection_dim = flp_downprojection_dim
        config.hlm_chat_template_prompt = flp_chat_template_prompt
        config.hlm_cache_hidden_states = config.flp_cache_hidden_states
    # For simple_cnn, set max_length from flp_max_length
    elif feature_extractor_type == "simple_cnn":
        config.scnn_max_length = flp_max_length

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
# Text forest job description (serializable for multiprocessing)
# ---------------------------------------------------------------------------

@dataclass
class TextForestJob:
    """A single text_forest experiment arm to run on a GPU."""
    dgp_idx: int
    repeat_idx: int
    arm_name: str
    fraction: float
    seed: int
    n_confounders_used: int
    n_confounders_total: int
    equation_mode: str
    # Serialized data for the worker
    sim_df_dict: Dict[str, Any]  # sim_df as dict for serialization
    subset_spec_dicts: List[Dict[str, Any]]  # specs as dicts
    confounder_cols: Optional[List[str]]
    dgp_stats: Dict[str, Any]  # pre-computed dgp stats for format_result
    feature_extractor_type: str = "frozen_llm_pooler"


def _build_text_forest_jobs(
    dgps: list,
    df: pd.DataFrame,
    num_repeats: int,
    confounder_fractions: List[float],
    equation_mode: str,
    output_path: Path,
    resume: bool,
    extractor_types: List[str] = None,
) -> List[TextForestJob]:
    """Build all text_forest jobs across DGPs, repeats, and extractor types."""
    if extractor_types is None:
        extractor_types = ["frozen_llm_pooler"]

    jobs = []
    for dgp_idx, dgp in enumerate(dgps):
        for repeat_idx in range(num_repeats):
            seed = 42 + dgp_idx * 1000 + repeat_idx
            sim_df = prepare_simulation_dataset(dgp, df, seed)

            # Pre-compute dgp_stats (avoids passing full DGP to workers)
            n_patients = len(sim_df)
            total_missing = 0
            for spec in dgp.confounder_specs:
                miss_col = f"explicit_conf_{spec.name}_missing"
                if miss_col in dgp.extracted_df.columns:
                    total_missing += dgp.extracted_df[miss_col].sum()
            missingness_rate = total_missing / (n_patients * len(dgp.confounder_specs)) if dgp.confounder_specs else 0
            dgp_stats = {
                "extraction_missingness_rate": float(missingness_rate),
                "simulated_treatment_rate": float(sim_df['treatment_indicator'].mean()),
                "simulated_outcome_rate": float(sim_df['outcome_indicator'].mean()),
                "true_ate": float(sim_df['true_ite_prob'].mean()),
                "true_ite_std": float(sim_df['true_ite_prob'].std()),
            }

            for extractor_type in extractor_types:
                for fraction in confounder_fractions:
                    n_subset = round(len(dgp.confounder_specs) * fraction)
                    # Include extractor type in arm name for multi-extractor runs
                    if len(extractor_types) > 1:
                        arm_name = f"text_forest_{extractor_type}_{fraction:.2f}"
                    else:
                        arm_name = f"text_forest_{fraction:.2f}"

                    if resume and is_result_done(output_path, dgp_idx, repeat_idx, arm_name):
                        continue

                    subset_specs = select_confounder_subset(
                        dgp.confounder_specs, n_subset, seed=seed
                    )
                    confounder_cols = get_confounder_cols(subset_specs, dgp.extracted_df)

                    jobs.append(TextForestJob(
                        dgp_idx=dgp_idx,
                        repeat_idx=repeat_idx,
                        arm_name=arm_name,
                        fraction=fraction,
                        seed=seed,
                        n_confounders_used=n_subset,
                        n_confounders_total=len(dgp.confounder_specs),
                        equation_mode=equation_mode,
                        sim_df_dict=sim_df.to_dict(orient='list'),
                        subset_spec_dicts=[asdict(s) for s in subset_specs] if subset_specs else [],
                        confounder_cols=confounder_cols,
                        dgp_stats=dgp_stats,
                        feature_extractor_type=extractor_type,
                    ))
    return jobs


def _run_text_forest_job(
    job: TextForestJob,
    device: str,
    output_path: Path,
    flp_model_name: str,
    flp_max_length: int,
    flp_downprojection_dim: int,
    flp_projection_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    n_folds: int,
    cf_n_estimators: int,
    cf_min_samples_leaf: int,
    hidden_state_cache=None,
    gpu_store=None,
    flp_chat_template_prompt: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Execute a single text_forest job. Returns formatted result or None on failure."""
    from oci.config import ExplicitConfounderSpec, CACHEABLE_EXTRACTOR_TYPES

    sim_df = pd.DataFrame(job.sim_df_dict)
    subset_specs = [ExplicitConfounderSpec(**d) for d in job.subset_spec_dicts] if job.subset_spec_dicts else None

    # Only pass cache/gpu_store for cacheable extractor types
    job_hidden_state_cache = hidden_state_cache if job.feature_extractor_type in CACHEABLE_EXTRACTOR_TYPES else None
    job_gpu_store = gpu_store if job.feature_extractor_type in CACHEABLE_EXTRACTOR_TYPES else None

    logger.info(
        f"  Running {job.arm_name} ({job.feature_extractor_type}, text + {job.n_confounders_used} confounders) "
        f"dgp={job.dgp_idx} repeat={job.repeat_idx} on {device}..."
    )
    try:
        result = run_text_forest_arm(
            df=sim_df,
            confounder_specs=subset_specs,
            confounder_cols=job.confounder_cols,
            seed=job.seed,
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
            hidden_state_cache=job_hidden_state_cache,
            gpu_store=job_gpu_store,
            flp_chat_template_prompt=flp_chat_template_prompt,
            feature_extractor_type=job.feature_extractor_type,
        )
        formatted = {
            "dgp_index": job.dgp_idx,
            "repeat_index": job.repeat_idx,
            "arm": job.arm_name,
            "uses_text": True,
            "n_confounders_used": job.n_confounders_used,
            "n_confounders_total": job.n_confounders_total,
            "confounder_fraction": job.fraction,
            "equation_mode": job.equation_mode,
            "metrics": result.get('metrics', {}),
            "n_samples": result.get('n_samples', len(sim_df)),
            "dgp_stats": job.dgp_stats,
        }
        save_result(formatted, output_path)
        ite_corr = result.get('metrics', {}).get('ite_corr', 'N/A')
        logger.info(f"    {job.arm_name} dgp={job.dgp_idx} repeat={job.repeat_idx} ITE corr: {ite_corr}")
        return formatted
    except Exception as e:
        logger.error(f"    {job.arm_name} dgp={job.dgp_idx} repeat={job.repeat_idx} Failed: {e}")
        traceback.print_exc()
        return None


def semisynthetic_worker_process_fn(
    device: str,
    job_queue: mp.Queue,
    progress_queue: mp.Queue,
    output_dir: str,
    cache_hash: str,
    cache_info: Optional[dict],
    use_gpu_cache: bool,
    # Model hyperparams passed as dict
    model_kwargs: Dict[str, Any],
    # Per-extractor-type cache info for multi-extractor support
    cache_info_by_type: Optional[Dict[str, dict]] = None,
):
    """Worker process for text_forest jobs on a single GPU (multiprocessing mode)."""
    from oci.models.hidden_state_cache import HiddenStateCache
    from oci.models.gpu_hidden_state_store import GPUHiddenStateStore

    output_path = Path(output_dir)
    torch.set_default_dtype(torch.float32)

    # Open per-extractor-type caches
    cache_registry = {}   # cache_hash -> HiddenStateCache
    gpu_store_registry = {}  # cache_hash -> GPUHiddenStateStore

    if cache_info_by_type:
        for et, ci in cache_info_by_type.items():
            ch = HiddenStateCache.compute_cache_hash(
                ci['model_name'], ci['max_length'], str(ci['parquet_file']), None,
                downprojection_dim=ci['downprojection_dim'],
                chat_template_prompt=ci.get('chat_template_prompt'),
                chunk_size=ci.get('chunk_size'),
                chunk_overlap=ci.get('chunk_overlap'),
                max_chunks=ci.get('max_chunks'),
            )
            cache = _open_cache_for_worker(ch, ci)
            cache_registry[ch] = cache
            if use_gpu_cache:
                store = load_single_gpu_store(cache, ci, device)
                if store is not None:
                    gpu_store_registry[ch] = store
    elif cache_info and cache_hash != "__no_cache__":
        # Legacy single-cache path
        cache = _open_cache_for_worker(cache_hash, cache_info)
        cache_registry[cache_hash] = cache
        if use_gpu_cache:
            store = load_single_gpu_store(cache, cache_info, device)
            if store is not None:
                gpu_store_registry[cache_hash] = store

    # Resolve cache for each job's extractor type
    def _resolve_cache(extractor_type):
        """Find the right cache/gpu_store for this extractor type."""
        from oci.config import CACHEABLE_EXTRACTOR_TYPES
        if extractor_type not in CACHEABLE_EXTRACTOR_TYPES:
            return None, None
        if cache_info_by_type and extractor_type in cache_info_by_type:
            ci = cache_info_by_type[extractor_type]
            ch = HiddenStateCache.compute_cache_hash(
                ci['model_name'], ci['max_length'], str(ci['parquet_file']), None,
                downprojection_dim=ci['downprojection_dim'],
                chat_template_prompt=ci.get('chat_template_prompt'),
                chunk_size=ci.get('chunk_size'),
                chunk_overlap=ci.get('chunk_overlap'),
                max_chunks=ci.get('max_chunks'),
            )
            gs = gpu_store_registry.get(ch)
            hsc = cache_registry.get(ch) if gs is None else None
            return hsc, gs
        # Legacy single-cache fallback
        if cache_registry:
            ch = list(cache_registry.keys())[0]
            gs = gpu_store_registry.get(ch)
            hsc = cache_registry.get(ch) if gs is None else None
            return hsc, gs
        return None, None

    logger.info(f"Text forest worker started on {device} (pid={os.getpid()}, "
                f"{len(cache_registry)} cache(s))")

    while True:
        try:
            job = job_queue.get(timeout=2)
        except Exception:
            break

        hsc, gs = _resolve_cache(job.feature_extractor_type)
        result = _run_text_forest_job(
            job=job,
            device=device,
            output_path=output_path,
            hidden_state_cache=hsc,
            gpu_store=gs,
            **model_kwargs,
        )
        progress_queue.put(("done" if result else "error", job.arm_name, result))

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Cleanup
    for store in gpu_store_registry.values():
        store.free()
    for cache in cache_registry.values():
        cache.close()

    logger.info(f"Text forest worker on {device} (pid={os.getpid()}) finished")


def semisynthetic_worker_thread(
    device: str,
    job_queue: queue.Queue,
    all_results: list,
    output_path: Path,
    lock: threading.Lock,
    progress_bar: tqdm,
    hidden_state_cache,
    gpu_store,
    model_kwargs: Dict[str, Any],
):
    """Worker thread for text_forest jobs on a single GPU (non-cached mode).

    In non-cached mode, hidden_state_cache and gpu_store are typically None.
    The model loads the LLM live per experiment.
    """
    while True:
        try:
            job = job_queue.get(timeout=1)
        except queue.Empty:
            break

        try:
            # Non-cacheable extractors should not get cache/gpu_store
            from oci.config import CACHEABLE_EXTRACTOR_TYPES
            job_hsc = hidden_state_cache if job.feature_extractor_type in CACHEABLE_EXTRACTOR_TYPES else None
            job_gs = gpu_store if job.feature_extractor_type in CACHEABLE_EXTRACTOR_TYPES else None
            result = _run_text_forest_job(
                job=job,
                device=device,
                output_path=output_path,
                hidden_state_cache=job_hsc,
                gpu_store=job_gs,
                **model_kwargs,
            )
            with lock:
                if result:
                    all_results.append(result)
                progress_bar.update(1)
                if result:
                    ite_corr = result.get('metrics', {}).get('ite_corr', 'N/A')
                    progress_bar.set_postfix_str(
                        f"{job.arm_name} ITE corr: {ite_corr:.3f}" if isinstance(ite_corr, float) else f"{job.arm_name}"
                    )
        except Exception as e:
            with lock:
                progress_bar.update(1)
                progress_bar.set_postfix_str(f"Error: {str(e)[:50]}")
            logger.error(f"Job {job.arm_name} failed: {e}")
            traceback.print_exc()
        finally:
            job_queue.task_done()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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
    flp_chat_template_prompt: Optional[str] = None,
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
    workers_per_gpu: str = "auto",
    # Feature extractor types for text_forest arms
    extractor_types: List[str] = None,
):
    """Main experiment runner.

    Phases:
    1. Start vLLM servers, generate all DGPs (confounders + equations)
    2. Shut down vLLM servers to free GPU VRAM
    3. Pre-cache frozen LLM hidden states (if --cache/--gpu-cache)
    4. Run CPU-only arms (confounder_forest, best_attainable)
    5. Run GPU arms (text_forest) in parallel across all devices
    """
    if confounder_fractions is None:
        confounder_fractions = [0.0, 0.25, 0.5, 0.75, 1.0]
    if devices is None:
        devices = ["cuda:0"]
    if extractor_types is None:
        extractor_types = ["frozen_llm_pooler"]

    use_cache = cache or gpu_cache
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

    # Load real dataset
    dp = Path(dataset_path)
    if dp.is_dir():
        parquet_file = dp / "dataset.parquet"
    else:
        parquet_file = dp
    df = pd.read_parquet(parquet_file)
    texts = df['clinical_text'].tolist()
    logger.info(f"Loaded dataset: {len(df)} patients from {parquet_file}")

    # =====================================================================
    # PHASE 1: Start vLLM servers and generate ALL DGPs
    # =====================================================================
    logger.info(f"\n{'='*60}")
    logger.info("PHASE 1: DGP generation (vLLM servers running)")
    logger.info(f"{'='*60}")

    vllm_server_urls = []
    vllm_server_procs = []
    if dgp_config.vllm_mode in ("server", "start_server"):
        servers = _ensure_vllm_servers(
            server_url=dgp_config.vllm_server_url,
            model_name=dgp_config.vllm_model_name,
            devices=devices,
            tensor_parallel_size=dgp_config.vllm_tensor_parallel_size,
            download_dir=dgp_config.vllm_download_dir,
            max_model_len=dgp_config.vllm_max_model_len,
        )
        vllm_server_urls = [url for _, url in servers]
        vllm_server_procs = [proc for proc, _ in servers if proc is not None]
        dgp_config.vllm_mode = "server"
        dgp_config.vllm_server_url = vllm_server_urls[0]

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
                server_urls=vllm_server_urls or None,
            )

    # Pre-generate ALL DGPs
    dgps = []
    for dgp_idx in range(num_dgps):
        logger.info(f"\nGenerating DGP {dgp_idx + 1}/{num_dgps}...")
        dgp = generate_dgp(
            dgp_config, texts, df, dataset_path, dgp_idx,
            cache_dir=str(output_path),
            confounders=shared_confounders,
            specs=shared_specs,
            extracted_df=shared_extracted_df,
            server_urls=vllm_server_urls or None,
        )
        save_dgp_metadata(dgp, output_path, dgp_idx)
        dgps.append(dgp)
    logger.info(f"All {num_dgps} DGP(s) generated")

    # =====================================================================
    # PHASE 2: Shut down vLLM servers to free GPU VRAM
    # =====================================================================
    if vllm_server_procs:
        logger.info(f"\n{'='*60}")
        logger.info("PHASE 2: Shutting down vLLM servers")
        logger.info(f"{'='*60}")
        _shutdown_vllm_servers(vllm_server_procs)
        vllm_server_procs = []

    # Save config (after DGP generation so dgp_config reflects final state)
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
        "extractor_types": extractor_types,
    }
    with open(output_path / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2, default=str)

    # =====================================================================
    # PHASE 3: Pre-cache frozen LLM hidden states (all GPUs free now)
    # =====================================================================
    hidden_state_cache = None
    gpu_store = None
    cache_info = None
    cache_info_by_type = {}

    # Only cache if at least one extractor type supports caching
    from oci.config import CACHEABLE_EXTRACTOR_TYPES
    has_cacheable_extractor = any(et in CACHEABLE_EXTRACTOR_TYPES for et in extractor_types)
    cacheable_in_use = [et for et in extractor_types if et in CACHEABLE_EXTRACTOR_TYPES]

    if use_cache and has_cacheable_extractor:
        logger.info(f"\n{'='*60}")
        logger.info("PHASE 3: Pre-caching LLM hidden states")
        logger.info(f"  Cacheable extractors: {cacheable_in_use}")
        logger.info(f"{'='*60}")

        # Build per-extractor-type cache_info (different extractors need different caches)
        # Use the first cacheable type as the "primary" for backward compat with single-cache paths
        cache_info_by_type = {}
        for et in cacheable_in_use:
            if et == "frozen_llm_pooler":
                cache_info_by_type[et] = dict(
                    parquet_file=parquet_file,
                    model_name=flp_model_name,
                    max_length=flp_max_length,
                    batch_size=batch_size,
                    downprojection_dim=flp_downprojection_dim,
                    chat_template_prompt=flp_chat_template_prompt,
                )
            elif et == "hierarchical_llm":
                # Use ExperimentConfig defaults for chunk params
                _hlm_chunk_size = ExperimentConfig.hlm_chunk_size
                _hlm_chunk_overlap = ExperimentConfig.hlm_chunk_overlap
                _hlm_max_chunks = ExperimentConfig.hlm_max_chunks
                cache_info_by_type[et] = dict(
                    parquet_file=parquet_file,
                    model_name=flp_model_name,
                    max_length=_hlm_chunk_size * _hlm_max_chunks,
                    batch_size=batch_size,
                    downprojection_dim=flp_downprojection_dim,
                    chat_template_prompt=flp_chat_template_prompt,
                    chunk_size=_hlm_chunk_size,
                    chunk_overlap=_hlm_chunk_overlap,
                    max_chunks=_hlm_max_chunks,
                )

        # Precompute each cache
        cache_by_type = {}
        for et, ci in cache_info_by_type.items():
            logger.info(f"  Precomputing cache for {et}...")
            cache_by_type[et] = precompute_single_cache(ci, devices)

        # Use the first cacheable type as the primary cache for backward compat
        primary_type = cacheable_in_use[0]
        cache_info = cache_info_by_type[primary_type]

        if gpu_cache:
            if workers_per_gpu == "1":
                gpu_store = load_single_gpu_store(cache_by_type[primary_type], cache_info, devices[0])
                if gpu_store is not None:
                    logger.info(f"GPU cache loaded: {gpu_store.num_samples} samples on {devices[0]}")
                else:
                    logger.info("GPU cache failed, falling back to disk cache")
                    hidden_state_cache = cache_by_type[primary_type]
            else:
                # Workers will load their own GPU stores
                for c in cache_by_type.values():
                    c.close()
        else:
            hidden_state_cache = cache_by_type[primary_type]

        # Close non-primary caches (workers reopen independently)
        for et, c in cache_by_type.items():
            if et != primary_type or (gpu_cache and workers_per_gpu != "1"):
                c.close()

    # =====================================================================
    # PHASE 4: Run CPU-only arms (confounder_forest, best_attainable)
    # =====================================================================
    logger.info(f"\n{'='*60}")
    logger.info("PHASE 4: Running CPU-only arms (confounder_forest, best_attainable)")
    logger.info(f"{'='*60}")

    all_results = []
    total_arms = 0

    for dgp_idx, dgp in enumerate(dgps):
        for repeat_idx in range(num_repeats):
            seed = 42 + dgp_idx * 1000 + repeat_idx
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
                    logger.info(f"  Skipping {arm_name} (0 confounders)")
                    continue

                logger.info(f"  Running {arm_name} ({n_subset} confounders) dgp={dgp_idx} repeat={repeat_idx}...")
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
                logger.info(f"  Running {arm_name} (all {len(dgp.confounder_specs)} confounders) dgp={dgp_idx} repeat={repeat_idx}...")
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

    logger.info(f"CPU arms complete: {total_arms} done")

    # =====================================================================
    # PHASE 5: Run text_forest arms in parallel across GPUs
    # =====================================================================
    logger.info(f"\n{'='*60}")
    logger.info("PHASE 5: Running text_forest arms (parallel across GPUs)")
    logger.info(f"{'='*60}")

    # Build all text_forest jobs
    text_forest_jobs = _build_text_forest_jobs(
        dgps=dgps,
        df=df,
        num_repeats=num_repeats,
        confounder_fractions=confounder_fractions,
        equation_mode=equation_mode,
        output_path=output_path,
        resume=resume,
        extractor_types=extractor_types,
    )

    if not text_forest_jobs:
        logger.info("No text_forest jobs to run (all done or skipped)")
    else:
        # Model kwargs shared by all jobs
        model_kwargs = dict(
            flp_model_name=flp_model_name,
            flp_max_length=flp_max_length,
            flp_downprojection_dim=flp_downprojection_dim,
            flp_projection_dim=flp_projection_dim,
            flp_chat_template_prompt=flp_chat_template_prompt,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            n_folds=n_folds,
            cf_n_estimators=cf_n_estimators,
            cf_min_samples_leaf=cf_min_samples_leaf,
        )

        logger.info(f"{len(text_forest_jobs)} text_forest jobs across {len(devices)} GPU(s) "
                     f"(workers-per-gpu: {workers_per_gpu})")

        if use_cache:
            # === MULTIPROCESSING PATH (cached mode) ===
            from oci.models.hidden_state_cache import HiddenStateCache

            # Serialize per-extractor-type cache info for workers
            serializable_cache_info_by_type = {}
            if cache_info_by_type:
                for et, ci in cache_info_by_type.items():
                    serializable_cache_info_by_type[et] = {
                        k: str(v) if isinstance(v, Path) else v
                        for k, v in ci.items()
                    }

            # Legacy single-cache hash (for backward compat arg)
            cache_hash = HiddenStateCache.compute_cache_hash(
                flp_model_name, flp_max_length, str(parquet_file), None,
                downprojection_dim=flp_downprojection_dim,
                chat_template_prompt=flp_chat_template_prompt,
            )
            serializable_cache_info = {
                k: str(v) if isinstance(v, Path) else v
                for k, v in cache_info.items()
            } if cache_info else {}

            # Close main-process cache handles (workers reopen independently)
            if hidden_state_cache is not None:
                hidden_state_cache.close()
                hidden_state_cache = None
            if gpu_store is not None:
                gpu_store.free()
                gpu_store = None

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            ctx = mp.get_context('spawn')
            job_queue = ctx.Queue()
            progress_queue = ctx.Queue()

            for job in text_forest_jobs:
                job_queue.put(job)

            # Spawn worker processes
            n_workers_per_gpu = resolve_workers_per_gpu(workers_per_gpu, devices[0], use_cache)
            processes = []
            for device in devices:
                for _ in range(n_workers_per_gpu):
                    p = ctx.Process(
                        target=semisynthetic_worker_process_fn,
                        args=(device, job_queue, progress_queue, str(output_path),
                              cache_hash, serializable_cache_info, gpu_cache,
                              model_kwargs),
                        kwargs=dict(cache_info_by_type=serializable_cache_info_by_type),
                        name=f"worker-{device}",
                    )
                    p.start()
                    processes.append(p)

            logger.info(f"Spawned {len(processes)} worker processes "
                         f"({n_workers_per_gpu} per GPU)")

            # Monitor progress
            progress_bar = tqdm(total=len(text_forest_jobs), desc="Text forest arms")
            completed = 0
            while completed < len(text_forest_jobs):
                alive = [p for p in processes if p.is_alive()]
                if not alive and completed < len(text_forest_jobs):
                    logger.error(f"All workers died with {len(text_forest_jobs) - completed} jobs remaining")
                    break

                try:
                    status, arm_name, result = progress_queue.get(timeout=5)
                    completed += 1
                    progress_bar.update(1)
                    if result:
                        all_results.append(result)
                        total_arms += 1
                        ite_corr = result.get('metrics', {}).get('ite_corr', 'N/A')
                        progress_bar.set_postfix_str(
                            f"{arm_name} ITE corr: {ite_corr:.3f}" if isinstance(ite_corr, float) else arm_name
                        )
                except Exception:
                    pass  # timeout, retry

            progress_bar.close()

            # Join workers
            for p in processes:
                p.join(timeout=30)
                if p.is_alive():
                    logger.warning(f"Worker {p.name} did not exit cleanly, terminating")
                    p.terminate()

        else:
            # === THREADING PATH (non-cached / live LLM mode) ===
            job_queue_t = queue.Queue()
            for job in text_forest_jobs:
                job_queue_t.put(job)

            lock = threading.Lock()
            progress_bar = tqdm(total=len(text_forest_jobs), desc="Text forest arms")

            threads = []
            for device in devices:
                t = threading.Thread(
                    target=semisynthetic_worker_thread,
                    args=(device, job_queue_t, all_results, output_path, lock,
                          progress_bar, hidden_state_cache, gpu_store, model_kwargs),
                    name=f"worker-{device}",
                )
                t.start()
                threads.append(t)

            for t in threads:
                t.join()

            progress_bar.close()
            # Count text_forest results added during threading
            total_arms = len(all_results)

    # =====================================================================
    # Aggregate results
    # =====================================================================
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
    parser.add_argument("--chat-template-prompt", type=str, default=None,
                        help="Chat template prompt for instruct models (default: None = disabled)")

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
    parser.add_argument("--workers-per-gpu", type=str, default="auto",
                        help="Concurrent text_forest workers per GPU: 'auto' or integer "
                             "(default: auto). Only effective with --cache/--gpu-cache; "
                             "non-cached mode always uses 1.")
    parser.add_argument("--extractor-types", nargs="+", default=["frozen_llm_pooler"],
                        help="Feature extractor types for text_forest arms "
                             "(default: frozen_llm_pooler). Multiple types create separate "
                             "arms per extractor.")

    args = parser.parse_args()

    # Validate --workers-per-gpu
    if args.workers_per_gpu != "auto":
        try:
            wpg = int(args.workers_per_gpu)
            if wpg < 1:
                parser.error("--workers-per-gpu must be >= 1")
        except ValueError:
            parser.error(f"--workers-per-gpu must be 'auto' or an integer, got '{args.workers_per_gpu}'")

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
        flp_chat_template_prompt=args.chat_template_prompt,
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
        workers_per_gpu=args.workers_per_gpu,
        extractor_types=args.extractor_types,
    )


if __name__ == "__main__":
    main()
