# oci/training/plasmode.py
"""Plasmode simulation experiments for sensitivity analysis."""

import logging
import random
import json
import gc
from pathlib import Path
from dataclasses import asdict
from typing import Optional, List, Tuple, Dict, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy import stats as scipy_stats
from joblib import Parallel, delayed

from ..config import AppliedInferenceConfig, PlasmodeExperimentConfig, PlasmodeConfig, normalize_feature_extractor_type
from ..models.causal_text import CausalText
from ..models.causal_text_forest import CausalTextForest
from ..models.hidden_state_cache import HiddenStateCache
from ..data import ClinicalTextDataset, collate_batch, CachedHiddenStateDataset, collate_cached_batch, prepare_cached_batch
from ..utils import cuda_cleanup, get_memory_info, set_seed


logger = logging.getLogger(__name__)


def run_plasmode_experiments(
    dataset: pd.DataFrame,
    applied_config: AppliedInferenceConfig,
    plasmode_config: PlasmodeExperimentConfig,
    output_path: Path,
    device: torch.device,
    cache=None,  # Kept for API compatibility
    num_repeats: int = 3,
    num_workers: int = 1,
    gpu_ids: Optional[List[int]] = None
) -> None:
    """
    Run plasmode sensitivity experiments.
    """
    logger.info("=" * 80)
    logger.info(f"PLASMODE SENSITIVITY EXPERIMENTS (Workers: {num_workers})")
    logger.info("=" * 80)

    train_df = dataset.copy()

    # Propensity trimming preprocessing (if enabled)
    if hasattr(plasmode_config, 'propensity_trimming') and plasmode_config.propensity_trimming.enabled:
        logger.info("=" * 80)
        logger.info("PROPENSITY-BASED DATASET TRIMMING FOR PLASMODE")
        logger.info("=" * 80)

        from .propensity_trimming import train_propensity_model_cv, trim_by_propensity

        # Train propensity model with CV to get out-of-sample scores
        train_df, propensity_training_log = train_propensity_model_cv(
            train_df, applied_config, device, num_workers, gpu_ids
        )

        # Save propensity model training log
        training_log_path = output_path.parent / "plasmode_propensity_trimming_training_log.csv"
        propensity_training_log.to_csv(training_log_path, index=False)
        logger.info(f"Propensity training log saved to: {training_log_path}")

        original_size = len(train_df)

        # Trim dataset
        train_df, trimming_stats = trim_by_propensity(
            train_df,
            plasmode_config.propensity_trimming.min_propensity,
            plasmode_config.propensity_trimming.max_propensity
        )

        logger.info(f"Plasmode base data trimmed: {original_size} -> {len(train_df)} "
                   f"({trimming_stats['removed_low']} below min, "
                   f"{trimming_stats['removed_high']} above max)")

        # Save trimming stats
        trimming_stats_path = output_path.parent / "plasmode_propensity_trimming_stats.json"
        with open(trimming_stats_path, 'w') as f:
            json.dump(trimming_stats, f, indent=2)
        logger.info(f"Trimming stats saved to: {trimming_stats_path}")

        logger.info("=" * 80)

    # Pre-compute and cache LLM hidden states for frozen_llm_pooler
    hidden_state_cache_config = None  # Serializable config for parallel workers
    # Check both generator and evaluator architectures
    for arch_label, arch in [("generator", plasmode_config.generator_architecture),
                              ("evaluator", plasmode_config.evaluator_architecture)]:
        feat_type = normalize_feature_extractor_type(
            getattr(arch, 'feature_extractor_type', 'frozen_llm_pooler')
        )
        if feat_type == "frozen_llm_pooler":
            flp_freeze = getattr(arch, 'flp_freeze_llm', True)
            flp_cache_enabled = getattr(arch, 'flp_cache_hidden_states', True)

            if flp_cache_enabled and flp_freeze:
                # Reset index for consistent cache indexing
                train_df = train_df.reset_index(drop=True)
                model_name = getattr(arch, 'flp_model_name', 'Qwen/Qwen3-0.6B-Base')
                max_length = getattr(arch, 'flp_max_length', 8192)
                dataset_path = applied_config.dataset_path

                flp_downprojection_dim = getattr(arch, 'flp_downprojection_dim', None)
                cache_dir = str(Path(dataset_path).parent / ".oci_cache")
                cache = HiddenStateCache(
                    cache_dir=cache_dir,
                    model_name=model_name,
                    max_length=max_length,
                    dataset_path=dataset_path,
                    downprojection_dim=flp_downprojection_dim,
                )

                all_texts = train_df[applied_config.text_column].tolist()
                if not cache.is_valid(len(train_df)):
                    logger.info(f"Pre-computing LLM hidden states for plasmode ({arch_label} arch)...")
                    batch_size = getattr(plasmode_config.generator_training, 'batch_size', 8)
                    try:
                        # Use multi-GPU precomputation when multiple GPUs available
                        precompute_devices = [device]
                        if gpu_ids and device.type == "cuda":
                            precompute_devices = [torch.device(f"cuda:{i}") for i in gpu_ids]
                        if len(precompute_devices) > 1:
                            logger.info(f"Using {len(precompute_devices)} GPUs for parallel precomputation")
                            cache.precompute_multi_gpu(
                                all_texts, precompute_devices, batch_size=batch_size
                            )
                        else:
                            cache.precompute(all_texts, device, batch_size=batch_size)
                    except Exception as e:
                        logger.warning(f"Hidden state caching failed: {e}. Falling back to non-cached mode.")
                        cache = None
                else:
                    logger.info("Reusing existing hidden state cache for plasmode")

                if cache is not None:
                    # Store serializable config for parallel workers
                    cache.open()
                    hidden_state_cache_config = {
                        'cache_dir': cache_dir,
                        'model_name': model_name,
                        'max_length': max_length,
                        'dataset_path': dataset_path,
                        'hidden_size': cache.hidden_size,
                        'downprojection_dim': flp_downprojection_dim,
                    }
                    cache.close()
                break  # Only need one cache (same texts for both architectures)
            elif flp_cache_enabled and not flp_freeze:
                logger.warning(
                    "flp_cache_hidden_states=True but flp_freeze_llm=False. "
                    "Caching is only supported with frozen LLM. Skipping cache."
                )

    logger.info(f"Using {len(train_df)} samples for plasmode generation base")
    logger.info(f"Running {len(plasmode_config.plasmode_scenarios)} scenarios x {num_repeats} repeats")

    # Dataset saving setup
    save_datasets = getattr(plasmode_config, 'save_datasets', False)
    dataset_dir = None
    if save_datasets:
        dataset_dir = output_path.parent / "simulated_datasets"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Simulated datasets will be saved to: {dataset_dir}")

    # Prepare all tasks
    tasks = []
    for scenario_idx, scenario in enumerate(plasmode_config.plasmode_scenarios):
        logger.info(f"\n{'=' * 80}")
        logger.info(f"SCENARIO {scenario_idx + 1}/{len(plasmode_config.plasmode_scenarios)}")
        logger.info(f"  Mode: {scenario.generation_mode}")
        logger.info(f"  Target ATE (prob): {scenario.target_ate_prob}")
        logger.info(f"{'=' * 80}")

        for repeat_idx in range(num_repeats):
            if gpu_ids and device.type == "cuda":
                task_global_idx = len(tasks)
                device_id = gpu_ids[task_global_idx % len(gpu_ids)]
                task_device = torch.device(f"cuda:{device_id}")
            else:
                # MPS and CPU are single-device; ignore gpu_ids
                task_device = device

            tasks.append({
                'scenario_idx': scenario_idx,
                'scenario': scenario,
                'repeat_idx': repeat_idx,
                'train_df': train_df,
                'applied_config': applied_config,
                'plasmode_config': plasmode_config,
                'device': task_device,
                'dataset_dir': dataset_dir,
                'hidden_state_cache_config': hidden_state_cache_config,
            })

    logger.info(f"Starting {len(tasks)} experiments on {num_workers} workers...")

    results = Parallel(n_jobs=num_workers)(
        delayed(_worker_wrapper)(task) for task in tasks
    )

    # Aggregate results
    all_results = []
    all_training_logs = []

    for res in results:
        if res is not None:
            metrics, logs = res
            all_results.append(metrics)
            all_training_logs.extend(logs)

    # Save aggregated training logs
    if all_training_logs:
        log_path = output_path.parent / "plasmode_training_log_aggregate.csv"
        pd.DataFrame(all_training_logs).to_csv(log_path, index=False)
        logger.info(f"Aggregated plasmode training logs saved to: {log_path}")

    # Save results summary
    if all_results:
        results_df = pd.DataFrame(all_results)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)

        logger.info(f"\n{'=' * 80}")
        logger.info("PLASMODE EXPERIMENTS COMPLETE")
        logger.info(f"Results saved to: {output_path}")
        logger.info(f"Total experiments: {len(results_df)}")
        logger.info(f"{'=' * 80}")

        summary = results_df.groupby('generation_mode').agg({
            'ate_bias_prob': ['mean', 'std'],
            'ate_rmse_prob': ['mean', 'std'],
            'ite_correlation_prob': ['mean', 'std'],
            'ite_spearman_correlation_prob': ['mean', 'std'],
        }).round(4)

        logger.info("\nSummary by generation mode (probability scale):")
        logger.info(f"\n{summary}")
    else:
        logger.error("No successful experiments completed")


def _worker_wrapper(task: Dict[str, Any]) -> Optional[Tuple[dict, List[Dict[str, Any]]]]:
    """Helper for parallel execution."""
    scenario_idx = task['scenario_idx']
    repeat_idx = task['repeat_idx']
    scenario = task['scenario']
    dataset_dir = task['dataset_dir']
    plasmode_config = task['plasmode_config']
    cache_config = task.get('hidden_state_cache_config')

    current_seed = random.randint(0, 10000)
    set_seed(current_seed)

    hyperparams = {
        'scenario_idx': scenario_idx,
        'repeat_idx': repeat_idx,
        'seed': current_seed,
        'generation_mode': scenario.generation_mode,
        **asdict(scenario),
    }

    save_dataset_path = None
    if dataset_dir:
        base_name = f"scenario_{scenario_idx}_repeat_{repeat_idx}_{scenario.generation_mode}"
        save_dataset_path = dataset_dir / f"{base_name}.parquet"

    # Open cache for this worker (each process needs its own handle)
    hidden_state_cache = None
    if cache_config is not None:
        hidden_state_cache = HiddenStateCache(
            cache_dir=cache_config['cache_dir'],
            model_name=cache_config['model_name'],
            max_length=cache_config['max_length'],
            dataset_path=cache_config['dataset_path'],
            downprojection_dim=cache_config.get('downprojection_dim'),
        )
        hidden_state_cache.open()
        hidden_state_cache.preload_to_ram()

    try:
        metrics, logs = _run_single_plasmode_experiment(
            train_df=task['train_df'],
            scenario=scenario,
            applied_config=task['applied_config'],
            plasmode_config=plasmode_config,
            device=task['device'],
            hyperparams=hyperparams,
            save_dataset_path=save_dataset_path,
            hidden_state_cache=hidden_state_cache,
        )

        metrics['scenario_idx'] = scenario_idx
        metrics['repeat_idx'] = repeat_idx
        metrics['generation_mode'] = scenario.generation_mode
        metrics['target_ate_prob'] = scenario.target_ate_prob
        metrics['train_fraction'] = plasmode_config.train_fraction

        return metrics, logs

    except Exception as e:
        logger.error(f"Scenario {scenario_idx} Repeat {repeat_idx} Failed: {e}", exc_info=True)
        return None
    finally:
        if hidden_state_cache is not None:
            hidden_state_cache.close()
        cuda_cleanup()


def _run_single_plasmode_experiment(
    train_df: pd.DataFrame,
    scenario: PlasmodeConfig,
    applied_config: AppliedInferenceConfig,
    plasmode_config: PlasmodeExperimentConfig,
    device: torch.device,
    hyperparams: Optional[Dict[str, Any]] = None,
    save_dataset_path: Optional[Path] = None,
    hidden_state_cache: Optional[HiddenStateCache] = None,
) -> Tuple[dict, List[Dict[str, Any]]]:
    """Run a single plasmode experiment."""

    train_fraction = getattr(plasmode_config, 'train_fraction', 0.8)
    seed = hyperparams.get('seed', 42) if hyperparams else 42

    # Ensure consistent indexing for cache
    train_df = train_df.reset_index(drop=True)

    # Split data
    train_split_df, eval_split_df = train_test_split(
        train_df, train_size=train_fraction, random_state=seed
    )

    # Track original indices for cache before resetting
    train_cache_indices = train_split_df.index.values if hidden_state_cache is not None else None
    eval_cache_indices = eval_split_df.index.values if hidden_state_cache is not None else None

    train_split_df = train_split_df.reset_index(drop=True)
    eval_split_df = eval_split_df.reset_index(drop=True)

    logger.info(f"Single-split: Training on {len(train_split_df)}, evaluating on {len(eval_split_df)}")

    # Step 1: Train generator (trained on real data with real outcome type)
    generator_outcome_type = getattr(applied_config, 'outcome_type', 'binary')
    generator, gen_history = _train_cnn_model(
        train_split_df,
        eval_split_df,
        applied_config,
        plasmode_config.generator_architecture,
        plasmode_config.generator_training,
        device,
        outcome_type=generator_outcome_type,
        hidden_state_cache=hidden_state_cache,
        train_indices=train_cache_indices,
        val_indices=eval_cache_indices
    )

    for entry in gen_history:
        entry['model_type'] = 'generator'
        entry['generation_mode'] = scenario.generation_mode
        entry['scenario_idx'] = hyperparams.get('scenario_idx', -1)
        entry['repeat_idx'] = hyperparams.get('repeat_idx', -1)

    # Step 2: Generate synthetic outcomes
    train_plasmode_df = _generate_plasmode_data(
        train_split_df, generator, scenario, applied_config, device,
        hidden_state_cache=hidden_state_cache,
        dataset_indices=train_cache_indices
    )
    eval_plasmode_df = _generate_plasmode_data(
        eval_split_df, generator, scenario, applied_config, device,
        hidden_state_cache=hidden_state_cache,
        dataset_indices=eval_cache_indices
    )

    train_plasmode_df['sim_split'] = 'train'
    eval_plasmode_df['sim_split'] = 'eval'

    # Step 3: Train evaluator on simulated data (uses scenario's outcome type)
    scenario_outcome_type = getattr(scenario, 'outcome_type', 'binary')
    evaluator, eval_history = _train_cnn_model(
        train_plasmode_df,
        eval_plasmode_df,
        applied_config,
        plasmode_config.evaluator_architecture,
        plasmode_config.evaluator_training,
        device,
        outcome_type=scenario_outcome_type,
        hidden_state_cache=hidden_state_cache,
        train_indices=train_cache_indices,
        val_indices=eval_cache_indices
    )

    for entry in eval_history:
        entry['model_type'] = 'evaluator'
        entry['generation_mode'] = scenario.generation_mode
        entry['scenario_idx'] = hyperparams.get('scenario_idx', -1)
        entry['repeat_idx'] = hyperparams.get('repeat_idx', -1)

    combined_history = gen_history + eval_history

    # Step 4: Generate predictions for eval split
    preds_dict = _predict_cnn_model(
        evaluator, eval_plasmode_df, applied_config, device,
        hidden_state_cache=hidden_state_cache,
        dataset_indices=eval_cache_indices
    )

    # Probability scale predictions only
    eval_plasmode_df['estimated_y0_prob'] = preds_dict['y0_prob']
    eval_plasmode_df['estimated_y1_prob'] = preds_dict['y1_prob']
    eval_plasmode_df['estimated_ite_prob'] = preds_dict['ite_prob']
    eval_plasmode_df['estimated_propensity_prob'] = preds_dict['propensity_prob']

    # Save dataset
    if save_dataset_path is not None:
        eval_plasmode_df.to_parquet(save_dataset_path, index=False)

    # Step 5: Evaluate (probability scale)
    metrics = _evaluate_plasmode_performance(
        eval_plasmode_df,
        scenario.target_ate_prob
    )

    logger.info(f"Experiment complete: ATE bias={metrics['ate_bias_prob']:.4f}, "
                f"ITE corr={metrics['ite_correlation_prob']:.4f}, "
                f"ITE rank corr={metrics['ite_spearman_correlation_prob']:.4f}")

    metrics['n_train'] = len(train_split_df)
    metrics['n_eval'] = len(eval_split_df)

    # Cleanup
    del generator, evaluator
    gc.collect()
    cuda_cleanup()

    return metrics, combined_history


def _train_cnn_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    applied_config: AppliedInferenceConfig,
    arch_config,
    train_config,
    device: torch.device,
    outcome_type: str = "binary",
    hidden_state_cache: Optional[HiddenStateCache] = None,
    train_indices: Optional[np.ndarray] = None,
    val_indices: Optional[np.ndarray] = None
) -> Tuple[Union[CausalText, CausalTextForest], List[Dict[str, Any]]]:
    """Train a model with frozen LLM pooler, Causal Forest, or TF-IDF Forest.

    Dispatches to specialized trainers for model_type="causal_forest" or "tfidf_forest".
    """
    # Check for causal forest model type
    model_type = getattr(arch_config, 'model_type', 'dragonnet')
    if model_type == "causal_forest":
        return _train_causal_forest_model(
            train_df, val_df, applied_config, arch_config, train_config, device,
            outcome_type=outcome_type,
            hidden_state_cache=hidden_state_cache,
            train_indices=train_indices,
            val_indices=val_indices
        )

    if model_type == "tfidf_forest":
        return _train_tfidf_forest_model(
            train_df, val_df, applied_config, arch_config,
            outcome_type=outcome_type
        )

    # Get feature extractor type
    # Normalize feature extractor type
    feature_extractor_type = normalize_feature_extractor_type(
        getattr(arch_config, 'feature_extractor_type', 'frozen_llm_pooler')
    )

    model = CausalText(
        feature_extractor_type=feature_extractor_type,
        # Frozen LLM Pooler args
        flp_model_name=getattr(arch_config, 'flp_model_name', 'Qwen/Qwen3-0.6B-Base'),
        flp_max_length=getattr(arch_config, 'flp_max_length', 8192),
        flp_freeze_llm=getattr(arch_config, 'flp_freeze_llm', True),
        flp_gated_attention_dim=getattr(arch_config, 'flp_gated_attention_dim', 128),
        flp_projection_dim=getattr(arch_config, 'flp_projection_dim', 128),
        flp_dropout=getattr(arch_config, 'flp_dropout', 0.1),
        flp_gradient_checkpointing=getattr(arch_config, 'flp_gradient_checkpointing', True),
        flp_downprojection_dim=(
            None if hidden_state_cache is not None
            else getattr(arch_config, 'flp_downprojection_dim', None)
        ),
        flp_skip_llm=(hidden_state_cache is not None),
        flp_cached_hidden_size=(hidden_state_cache.hidden_size if hidden_state_cache is not None else 0),
        # Numeric feature args
        numeric_features_enabled=getattr(arch_config, 'numeric_features_enabled', False),
        numeric_embedding_dim=getattr(arch_config, 'numeric_embedding_dim', 32),
        numeric_magnitude_bins=getattr(arch_config, 'numeric_magnitude_bins', 8),
        numeric_type_categories=getattr(arch_config, 'numeric_type_categories', 10),
        # Causal head args
        causal_head_representation_dim=arch_config.causal_head_representation_dim,
        causal_head_hidden_outcome_dim=arch_config.causal_head_hidden_outcome_dim,
        device=str(device),
        model_type=arch_config.model_type,
        # R-Learner dual extractor mode
        rlearner_dual_extractors=getattr(arch_config, 'rlearner_dual_extractors', False),
        # Outcome type
        outcome_type=outcome_type,
    )

    # Frozen LLM Pooler uses pretrained tokenizer, no fit_tokenizer needed
    logger.info(f"Using Frozen LLM Pooler feature extractor: {getattr(arch_config, 'flp_model_name', 'Qwen/Qwen3-0.6B-Base')} "
               f"({'frozen' if getattr(arch_config, 'flp_freeze_llm', True) else 'trainable'})"
               f"{' (cached)' if hidden_state_cache is not None else ''}")

    # Create datasets
    if hidden_state_cache is not None and train_indices is not None:
        cache_hs = hidden_state_cache.hidden_states_array
        cache_mask = hidden_state_cache.attention_mask_array
        train_dataset = CachedHiddenStateDataset(
            data=train_df,
            text_column=applied_config.text_column,
            outcome_column=applied_config.outcome_column,
            treatment_column=applied_config.treatment_column,
            dataset_indices=train_indices,
            cache_hidden_states=cache_hs,
            cache_attention_masks=cache_mask,
        )
        val_dataset = CachedHiddenStateDataset(
            data=val_df,
            text_column=applied_config.text_column,
            outcome_column=applied_config.outcome_column,
            treatment_column=applied_config.treatment_column,
            dataset_indices=val_indices,
            cache_hidden_states=cache_hs,
            cache_attention_masks=cache_mask,
        )
        collate_fn = collate_cached_batch
    else:
        train_dataset = ClinicalTextDataset(
            data=train_df,
            text_column=applied_config.text_column,
            outcome_column=applied_config.outcome_column,
            treatment_column=applied_config.treatment_column
        )
        val_dataset = ClinicalTextDataset(
            data=val_df,
            text_column=applied_config.text_column,
            outcome_column=applied_config.outcome_column,
            treatment_column=applied_config.treatment_column
        )
        collate_fn = collate_batch

    use_cached_mode = hidden_state_cache is not None and train_indices is not None
    if use_cached_mode:
        dl_kwargs = dict(num_workers=2, persistent_workers=True, pin_memory=True)
    else:
        dl_kwargs = dict(num_workers=2, persistent_workers=True, pin_memory=True, prefetch_factor=2)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        **dl_kwargs
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        **dl_kwargs
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=1e-4
    )

    history = []
    best_val_loss = float('inf')
    best_model_state = None

    # Get gamma_rlearner and advanced training options from config
    gamma_rlearner = getattr(train_config, 'gamma_rlearner', 1.0)
    gamma_dr = getattr(train_config, 'gamma_dr', 1.0)
    stop_grad_propensity = getattr(train_config, 'stop_grad_propensity', False)
    attention_entropy_weight = getattr(train_config, 'attention_entropy_weight', 0.0)

    for epoch in range(train_config.epochs):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            batch['outcome'] = batch['outcome'].to(device)
            batch['treatment'] = batch['treatment'].to(device)

            prepare_cached_batch(batch, device, hidden_state_cache)

            optimizer.zero_grad()
            losses = model.train_step(
                batch,
                alpha_propensity=train_config.alpha_propensity,
                beta_targreg=train_config.beta_targreg,
                gamma_rlearner=gamma_rlearner,
                gamma_dr=gamma_dr,
                stop_grad_propensity=stop_grad_propensity,
                attention_entropy_weight=attention_entropy_weight,
            )
            losses['loss'].backward()
            optimizer.step()
            epoch_loss += losses['loss'].item()

        train_loss = epoch_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch['outcome'] = batch['outcome'].to(device)
                batch['treatment'] = batch['treatment'].to(device)

                prepare_cached_batch(batch, device, hidden_state_cache)

                losses = model.train_step(
                    batch,
                    alpha_propensity=train_config.alpha_propensity,
                    beta_targreg=train_config.beta_targreg,
                    gamma_rlearner=gamma_rlearner,
                    gamma_dr=gamma_dr,
                    stop_grad_propensity=stop_grad_propensity,
                    attention_entropy_weight=attention_entropy_weight,
                )
                val_loss += losses['loss'].item()

        val_loss = val_loss / len(val_loader)

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

    if best_model_state:
        model.load_state_dict(best_model_state)

    return model, history


class TfidfForestWrapper:
    """Lightweight wrapper for TF-IDF + CausalForest, compatible with plasmode predict API."""

    def __init__(self, vectorizer, causal_forest, prop_rf, outcome_rf, outcome_type="binary"):
        self.vectorizer = vectorizer
        self.causal_forest = causal_forest
        self.prop_rf = prop_rf
        self.outcome_rf = outcome_rf
        self.outcome_type = outcome_type

    def predict_from_texts(self, texts):
        """Predict from raw texts. Returns dict with y0_prob, y1_prob, propensity_prob, ite_prob."""
        X = self.vectorizer.transform(texts).toarray()
        cf_preds = self.causal_forest.predict(X, return_ci=False)
        tau = cf_preds['tau_pred']
        propensity = self.prop_rf.predict_proba(X)[:, 1]
        if self.outcome_type == "continuous":
            outcome = self.outcome_rf.predict(X)
        else:
            outcome = self.outcome_rf.predict_proba(X)[:, 1]
        y0 = outcome - propensity * tau
        y1 = outcome + (1 - propensity) * tau
        if self.outcome_type == "binary":
            y0 = np.clip(y0, 0, 1)
            y1 = np.clip(y1, 0, 1)
        return {
            'y0_prob': y0,
            'y1_prob': y1,
            'propensity_prob': propensity,
            'ite_prob': tau
        }


def _train_tfidf_forest_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    applied_config: AppliedInferenceConfig,
    arch_config,
    outcome_type: str = "binary"
) -> Tuple[TfidfForestWrapper, List[Dict[str, Any]]]:
    """Train a TF-IDF + CausalForest model for plasmode experiments.

    No neural network, no GPU. Returns a TfidfForestWrapper.
    """
    from ..config import TfidfForestConfig
    from ..models.causal_forest_head import CausalForestHead
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    tfidf_config = getattr(arch_config, 'tfidf_forest', TfidfForestConfig())

    # Combine train + val
    combined_df = pd.concat([train_df, val_df])
    texts = combined_df[applied_config.text_column].tolist()
    T = combined_df[applied_config.treatment_column].values
    Y = combined_df[applied_config.outcome_column].values

    # TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=tfidf_config.max_features,
        ngram_range=(tfidf_config.ngram_range_min, tfidf_config.ngram_range_max),
        min_df=tfidf_config.min_df,
        max_df=tfidf_config.max_df,
        sublinear_tf=tfidf_config.sublinear_tf,
        dtype=np.float32
    )
    X = vectorizer.fit_transform(texts).toarray()
    logger.info(f"TF-IDF features: {X.shape[1]} (vocab: {len(vectorizer.vocabulary_)})")

    # Fit causal forest
    forest = CausalForestHead(
        n_estimators=tfidf_config.n_estimators,
        max_depth=tfidf_config.max_depth,
        min_samples_leaf=tfidf_config.min_samples_leaf,
        max_features=tfidf_config.max_features_forest,
        honest=tfidf_config.honest,
        inference=tfidf_config.inference,
        random_state=42
    )
    forest.fit(X, T, Y)

    # Nuisance models
    prop_rf = RandomForestClassifier(
        n_estimators=max(50, tfidf_config.n_estimators // 2),
        max_depth=tfidf_config.max_depth,
        min_samples_leaf=tfidf_config.min_samples_leaf,
        random_state=42, n_jobs=-1
    )
    prop_rf.fit(X, T)

    if outcome_type == "continuous":
        outcome_rf = RandomForestRegressor(
            n_estimators=max(50, tfidf_config.n_estimators // 2),
            max_depth=tfidf_config.max_depth,
            min_samples_leaf=tfidf_config.min_samples_leaf,
            random_state=42, n_jobs=-1
        )
    else:
        outcome_rf = RandomForestClassifier(
            n_estimators=max(50, tfidf_config.n_estimators // 2),
            max_depth=tfidf_config.max_depth,
            min_samples_leaf=tfidf_config.min_samples_leaf,
            random_state=42, n_jobs=-1
        )
    outcome_rf.fit(X, Y)

    model = TfidfForestWrapper(vectorizer, forest, prop_rf, outcome_rf, outcome_type=outcome_type)
    history = []  # No training epochs
    return model, history


def _train_causal_forest_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    applied_config: AppliedInferenceConfig,
    arch_config,
    train_config,
    device: torch.device,
    outcome_type: str = "binary",
    hidden_state_cache: Optional[HiddenStateCache] = None,
    train_indices: Optional[np.ndarray] = None,
    val_indices: Optional[np.ndarray] = None
) -> Tuple[CausalTextForest, List[Dict[str, Any]]]:
    """Train a CausalTextForest model for plasmode experiments.

    Two-stage approach:
        1. Train neural feature extractor with propensity + outcome losses
           (optionally with R-learner loss for representation training)
        2. Train causal forest on extracted features
    """
    feature_extractor_type = normalize_feature_extractor_type(
        getattr(arch_config, 'feature_extractor_type', 'frozen_llm_pooler')
    )

    # Get causal forest config
    cf_config = getattr(arch_config, 'causal_forest', None)
    if cf_config is None:
        cf_n_estimators = 100
        cf_max_depth = None
        cf_min_samples_leaf = 5
        cf_max_features = "sqrt"
        cf_honest = True
        cf_inference = True
        cf_use_rlearner_representation = False
        cf_gamma_rlearner = 1.0
    else:
        cf_n_estimators = getattr(cf_config, 'n_estimators', 100)
        cf_max_depth = getattr(cf_config, 'max_depth', None)
        cf_min_samples_leaf = getattr(cf_config, 'min_samples_leaf', 5)
        cf_max_features = getattr(cf_config, 'max_features', "sqrt")
        cf_honest = getattr(cf_config, 'honest', True)
        cf_inference = getattr(cf_config, 'inference', True)
        cf_use_rlearner_representation = getattr(cf_config, 'use_rlearner_representation', False)
        cf_gamma_rlearner = getattr(cf_config, 'gamma_rlearner', 1.0)

    model = CausalTextForest(
        feature_extractor_type=feature_extractor_type,
        # Frozen LLM Pooler args
        flp_model_name=getattr(arch_config, 'flp_model_name', 'Qwen/Qwen3-0.6B-Base'),
        flp_max_length=getattr(arch_config, 'flp_max_length', 8192),
        flp_freeze_llm=getattr(arch_config, 'flp_freeze_llm', True),
        flp_gated_attention_dim=getattr(arch_config, 'flp_gated_attention_dim', 128),
        flp_projection_dim=getattr(arch_config, 'flp_projection_dim', 128),
        flp_dropout=getattr(arch_config, 'flp_dropout', 0.1),
        flp_gradient_checkpointing=getattr(arch_config, 'flp_gradient_checkpointing', True),
        flp_downprojection_dim=(
            None if hidden_state_cache is not None
            else getattr(arch_config, 'flp_downprojection_dim', None)
        ),
        flp_skip_llm=(hidden_state_cache is not None),
        flp_cached_hidden_size=(hidden_state_cache.hidden_size if hidden_state_cache is not None else 0),
        # Head args
        representation_dim=getattr(arch_config, 'causal_head_representation_dim', 128),
        hidden_dim=getattr(arch_config, 'causal_head_hidden_outcome_dim', 64),
        dropout=getattr(arch_config, 'causal_head_dropout', 0.2),
        # Causal forest args
        cf_n_estimators=cf_n_estimators,
        cf_max_depth=cf_max_depth,
        cf_min_samples_leaf=cf_min_samples_leaf,
        cf_max_features=cf_max_features,
        cf_honest=cf_honest,
        cf_inference=cf_inference,
        cf_use_rlearner_representation=cf_use_rlearner_representation,
        cf_gamma_rlearner=cf_gamma_rlearner,
        # Numeric feature args
        numeric_features_enabled=getattr(arch_config, 'numeric_features_enabled', False),
        numeric_embedding_dim=getattr(arch_config, 'numeric_embedding_dim', 32),
        numeric_magnitude_bins=getattr(arch_config, 'numeric_magnitude_bins', 8),
        numeric_type_categories=getattr(arch_config, 'numeric_type_categories', 10),
        device=str(device),
        outcome_type=outcome_type
    )

    logger.info(f"Using CausalTextForest with {feature_extractor_type.upper()} extractor"
                f"{' (cached)' if hidden_state_cache is not None else ''}")

    # Create datasets
    if hidden_state_cache is not None and train_indices is not None:
        cache_hs = hidden_state_cache.hidden_states_array
        cache_mask = hidden_state_cache.attention_mask_array
        train_dataset = CachedHiddenStateDataset(
            data=train_df,
            text_column=applied_config.text_column,
            outcome_column=applied_config.outcome_column,
            treatment_column=applied_config.treatment_column,
            dataset_indices=train_indices,
            cache_hidden_states=cache_hs,
            cache_attention_masks=cache_mask,
        )
        val_dataset = CachedHiddenStateDataset(
            data=val_df,
            text_column=applied_config.text_column,
            outcome_column=applied_config.outcome_column,
            treatment_column=applied_config.treatment_column,
            dataset_indices=val_indices,
            cache_hidden_states=cache_hs,
            cache_attention_masks=cache_mask,
        )
        collate_fn = collate_cached_batch
    else:
        train_dataset = ClinicalTextDataset(
            data=train_df,
            text_column=applied_config.text_column,
            outcome_column=applied_config.outcome_column,
            treatment_column=applied_config.treatment_column
        )
        val_dataset = ClinicalTextDataset(
            data=val_df,
            text_column=applied_config.text_column,
            outcome_column=applied_config.outcome_column,
            treatment_column=applied_config.treatment_column
        )
        collate_fn = collate_batch

    use_cached_mode = hidden_state_cache is not None and train_indices is not None
    if use_cached_mode:
        dl_kwargs = dict(num_workers=2, persistent_workers=True, pin_memory=True)
    else:
        dl_kwargs = dict(num_workers=2, persistent_workers=True, pin_memory=True, prefetch_factor=2)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        **dl_kwargs
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        **dl_kwargs
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=getattr(train_config, 'weight_decay', 0.01)
    )

    history = []
    best_val_loss = float('inf')
    best_model_state = None

    # Training options
    alpha_propensity = train_config.alpha_propensity
    stop_grad_propensity = getattr(train_config, 'stop_grad_propensity', False)
    label_smoothing = getattr(train_config, 'label_smoothing', 0.0)
    gamma_rlearner = cf_gamma_rlearner if model.use_rlearner_representation else 0.0

    # Stage 1: Train representation
    for epoch in range(train_config.epochs):
        model.train()
        epoch_loss = 0.0
        train_r_loss = 0.0

        for batch in train_loader:
            batch['outcome'] = batch['outcome'].to(device)
            batch['treatment'] = batch['treatment'].to(device)

            prepare_cached_batch(batch, device, hidden_state_cache)

            optimizer.zero_grad()
            losses = model.train_representation_step(
                batch,
                alpha_propensity=alpha_propensity,
                gamma_rlearner=gamma_rlearner,
                label_smoothing=label_smoothing,
                stop_grad_propensity=stop_grad_propensity,
            )
            losses['loss'].backward()
            optimizer.step()
            epoch_loss += losses['loss'].item()
            train_r_loss += losses.get('r_loss', torch.tensor(0.0)).item()

        train_loss = epoch_loss / len(train_loader)
        train_r_loss = train_r_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch['outcome'] = batch['outcome'].to(device)
                batch['treatment'] = batch['treatment'].to(device)

                prepare_cached_batch(batch, device, hidden_state_cache)

                losses = model.train_representation_step(
                    batch,
                    alpha_propensity=alpha_propensity,
                    gamma_rlearner=gamma_rlearner,
                    stop_grad_propensity=stop_grad_propensity,
                )
                val_loss += losses['loss'].item()

        val_loss = val_loss / len(val_loader)

        epoch_log = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
        }
        if model.use_rlearner_representation:
            epoch_log['train_r_loss'] = train_r_loss
        history.append(epoch_log)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Restore best model state
    if best_model_state:
        model.load_state_dict(best_model_state)
        model.to(device)

    # Stage 2: Train causal forest on extracted features
    combined_df = pd.concat([train_df, val_df])
    if hidden_state_cache is not None and train_indices is not None and val_indices is not None:
        combined_indices = np.concatenate([train_indices, val_indices])
        combined_dataset = CachedHiddenStateDataset(
            data=combined_df,
            text_column=applied_config.text_column,
            outcome_column=applied_config.outcome_column,
            treatment_column=applied_config.treatment_column,
            dataset_indices=combined_indices,
            cache_hidden_states=hidden_state_cache.hidden_states_array,
            cache_attention_masks=hidden_state_cache.attention_mask_array,
        )
        combined_collate_fn = collate_cached_batch
    else:
        combined_dataset = ClinicalTextDataset(
            data=combined_df,
            text_column=applied_config.text_column,
            outcome_column=applied_config.outcome_column,
            treatment_column=applied_config.treatment_column
        )
        combined_collate_fn = collate_fn

    if hidden_state_cache is not None:
        dl_kwargs_combined = dict(num_workers=2, persistent_workers=True, pin_memory=True)
    else:
        dl_kwargs_combined = dict(num_workers=2, persistent_workers=True, pin_memory=True, prefetch_factor=2)
    combined_loader = DataLoader(
        combined_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        collate_fn=combined_collate_fn,
        **dl_kwargs_combined
    )

    # Hidden states already in DataLoader batches; prepare_cached_batch called in extract_features
    combined_T = combined_df[applied_config.treatment_column].values
    combined_Y = combined_df[applied_config.outcome_column].values
    model.train_causal_forest(combined_loader, combined_T, combined_Y)

    return model, history


def _generate_plasmode_data(
    df: pd.DataFrame,
    generator: Union[CausalText, CausalTextForest],
    scenario: PlasmodeConfig,
    applied_config: AppliedInferenceConfig,
    device: torch.device,
    hidden_state_cache: Optional[HiddenStateCache] = None,
    dataset_indices: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """Generate synthetic outcomes using the generator model."""

    plasmode_df = df.copy()

    # Get features from generator
    if hidden_state_cache is not None and dataset_indices is not None:
        dataset = CachedHiddenStateDataset(
            data=df,
            text_column=applied_config.text_column,
            outcome_column=applied_config.outcome_column,
            treatment_column=applied_config.treatment_column,
            dataset_indices=dataset_indices,
            cache_hidden_states=hidden_state_cache.hidden_states_array,
            cache_attention_masks=hidden_state_cache.attention_mask_array,
        )
        collate_fn = collate_cached_batch
    else:
        dataset = ClinicalTextDataset(
            data=df,
            text_column=applied_config.text_column,
            outcome_column=applied_config.outcome_column,
            treatment_column=applied_config.treatment_column
        )
        collate_fn = collate_batch

    dl_kwargs = dict(num_workers=2, persistent_workers=True, pin_memory=True) if (hidden_state_cache is not None) else {}
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn,
        **dl_kwargs
    )

    generator.eval()
    all_features = []

    with torch.no_grad():
        for batch in loader:
            prepare_cached_batch(batch, device, hidden_state_cache)

            # get_features accepts both text lists and batch dicts
            if 'cached_hidden_states' in batch:
                batch['texts'] = batch.get('texts', [])
                features = generator.get_features(batch)
            else:
                texts = batch['texts']
                features = generator.get_features(texts)
            all_features.append(features.cpu().numpy())

    confounder_features = np.concatenate(all_features, axis=0)

    # Generate synthetic ITEs based on scenario
    np.random.seed(42)

    # Convert target ATE from probability scale to logit scale for simulation
    # Use baseline control outcome rate to compute approximate logit ITE
    p0 = scenario.baseline_control_outcome_rate
    p1 = min(0.99, max(0.01, p0 + scenario.target_ate_prob))  # Clamp to valid range
    target_ate_logit = np.log(p1 / (1 - p1)) - np.log(p0 / (1 - p0))

    if scenario.generation_mode == "phi_linear":
        # Simple linear ITE based on features
        weights = np.random.randn(confounder_features.shape[1]) * 0.1
        base_ite = confounder_features @ weights
        base_ite = base_ite * scenario.ite_heterogeneity_scale
        ite_logit = base_ite + target_ate_logit

    else:
        # Default: constant ATE
        ite_logit = np.full(len(df), target_ate_logit)

    # Determine outcome type for this scenario
    scenario_outcome_type = getattr(scenario, 'outcome_type', 'binary')
    treatments = df[applied_config.treatment_column].values

    if scenario_outcome_type == "continuous":
        # Continuous outcomes: use logit-space values + Gaussian noise
        y0_logit = np.random.randn(len(df)) * scenario.outcome_heterogeneity_scale
        y0_logit += np.log(scenario.baseline_control_outcome_rate / (1 - scenario.baseline_control_outcome_rate))
        y1_logit = y0_logit + ite_logit

        # For continuous, the "value" is the logit itself (not passed through sigmoid)
        true_y0 = y0_logit
        true_y1 = y1_logit
        true_ite = true_y1 - true_y0

        # Observed outcome = true potential outcome + Gaussian noise
        observed_value = np.where(treatments == 1, true_y1, true_y0)
        noise_scale = scenario.outcome_heterogeneity_scale * 0.1  # Small noise
        observed_outcome = observed_value + np.random.randn(len(df)) * noise_scale

        plasmode_df[applied_config.outcome_column] = observed_outcome
        plasmode_df['true_y0_prob'] = true_y0
        plasmode_df['true_y1_prob'] = true_y1
        plasmode_df['true_ite_prob'] = true_ite
    else:
        # Binary outcomes: logit -> sigmoid -> Bernoulli
        y0_logit = np.random.randn(len(df)) * scenario.outcome_heterogeneity_scale
        y0_logit += np.log(scenario.baseline_control_outcome_rate / (1 - scenario.baseline_control_outcome_rate))
        y1_logit = y0_logit + ite_logit

        y0_prob = 1 / (1 + np.exp(-y0_logit))
        y1_prob = 1 / (1 + np.exp(-y1_logit))

        observed_prob = np.where(treatments == 1, y1_prob, y0_prob)
        observed_outcome = (np.random.rand(len(df)) < observed_prob).astype(float)

        plasmode_df[applied_config.outcome_column] = observed_outcome
        plasmode_df['true_y0_prob'] = y0_prob
        plasmode_df['true_y1_prob'] = y1_prob
        plasmode_df['true_ite_prob'] = y1_prob - y0_prob

    return plasmode_df


def _predict_cnn_model(
    model: Union[CausalText, CausalTextForest],
    df: pd.DataFrame,
    applied_config: AppliedInferenceConfig,
    device: torch.device,
    hidden_state_cache: Optional[HiddenStateCache] = None,
    dataset_indices: Optional[np.ndarray] = None
) -> dict:
    """Generate predictions from CausalText, CausalTextForest, or TfidfForestWrapper model."""

    # Handle TfidfForestWrapper (no neural network, no DataLoader)
    if isinstance(model, TfidfForestWrapper):
        texts = df[applied_config.text_column].tolist()
        return model.predict_from_texts(texts)

    # Handle CausalTextForest separately (uses different prediction API)
    if isinstance(model, CausalTextForest):
        if hidden_state_cache is not None and dataset_indices is not None:
            forest_dataset = CachedHiddenStateDataset(
                data=df,
                text_column=applied_config.text_column,
                outcome_column=applied_config.outcome_column,
                treatment_column=applied_config.treatment_column,
                dataset_indices=dataset_indices,
                cache_hidden_states=hidden_state_cache.hidden_states_array,
                cache_attention_masks=hidden_state_cache.attention_mask_array,
            )
            forest_collate_fn = collate_cached_batch
        else:
            forest_dataset = ClinicalTextDataset(
                data=df,
                text_column=applied_config.text_column,
                outcome_column=applied_config.outcome_column,
                treatment_column=applied_config.treatment_column
            )
            forest_collate_fn = collate_batch
        dl_kwargs_forest = dict(num_workers=2, persistent_workers=True, pin_memory=True) if (hidden_state_cache is not None) else {}
        forest_loader = DataLoader(
            forest_dataset,
            batch_size=32,
            shuffle=False,
            collate_fn=forest_collate_fn,
            **dl_kwargs_forest
        )
        # Hidden states already in DataLoader batches; prepare_cached_batch called in extract_features
        preds = model.predict(forest_loader, return_ci=False)
        return {
            'y0_prob': preds['pred_y0_prob'],
            'y1_prob': preds['pred_y1_prob'],
            'propensity_prob': preds['pred_propensity_prob'],
            'ite_prob': preds['pred_ite_prob']
        }

    # CausalText prediction path
    if hidden_state_cache is not None and dataset_indices is not None:
        dataset = CachedHiddenStateDataset(
            data=df,
            text_column=applied_config.text_column,
            outcome_column=applied_config.outcome_column,
            treatment_column=applied_config.treatment_column,
            dataset_indices=dataset_indices,
            cache_hidden_states=hidden_state_cache.hidden_states_array,
            cache_attention_masks=hidden_state_cache.attention_mask_array,
        )
        collate_fn = collate_cached_batch
    else:
        dataset = ClinicalTextDataset(
            data=df,
            text_column=applied_config.text_column,
            outcome_column=applied_config.outcome_column,
            treatment_column=applied_config.treatment_column
        )
        collate_fn = collate_batch

    dl_kwargs = dict(num_workers=2, persistent_workers=True, pin_memory=True) if (hidden_state_cache is not None) else {}
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn,
        **dl_kwargs
    )

    model.eval()
    all_y0 = []
    all_y1 = []
    all_prop = []

    with torch.no_grad():
        for batch in loader:
            prepare_cached_batch(batch, device, hidden_state_cache)

            preds = model.predict(batch)
            all_y0.append(preds['y0_logit'].cpu().numpy())
            all_y1.append(preds['y1_logit'].cpu().numpy())
            all_prop.append(preds['t_logit'].cpu().numpy())

    y0_logit = np.concatenate(all_y0)
    y1_logit = np.concatenate(all_y1)
    prop_logit = np.concatenate(all_prop)

    outcome_type = getattr(model, 'outcome_type', 'binary')

    if outcome_type == "continuous":
        # For continuous, logits ARE the predictions (no sigmoid)
        y0_prob = y0_logit
        y1_prob = y1_logit
    else:
        # Convert to probabilities using sigmoid
        y0_prob = 1.0 / (1.0 + np.exp(-y0_logit))
        y1_prob = 1.0 / (1.0 + np.exp(-y1_logit))

    # Propensity is always binary
    propensity_prob = 1.0 / (1.0 + np.exp(-prop_logit))
    ite_prob = y1_prob - y0_prob

    return {
        'y0_prob': y0_prob,
        'y1_prob': y1_prob,
        'propensity_prob': propensity_prob,
        'ite_prob': ite_prob
    }


def _evaluate_plasmode_performance(
    df: pd.DataFrame,
    target_ate_prob: float
) -> dict:
    """Evaluate plasmode performance on probability scale."""

    # Probability scale evaluation
    true_ite_prob = df['true_ite_prob'].values
    estimated_ite_prob = df['estimated_ite_prob'].values
    true_ate_prob = true_ite_prob.mean()
    estimated_ate_prob = estimated_ite_prob.mean()

    ate_bias_prob = estimated_ate_prob - true_ate_prob
    ate_rmse_prob = np.sqrt((estimated_ate_prob - true_ate_prob) ** 2)

    # ITE correlation (probability scale)
    if np.std(true_ite_prob) > 0 and np.std(estimated_ite_prob) > 0:
        ite_correlation_prob = np.corrcoef(true_ite_prob, estimated_ite_prob)[0, 1]
        ite_spearman_correlation_prob = scipy_stats.spearmanr(true_ite_prob, estimated_ite_prob)[0]
    else:
        ite_correlation_prob = 0.0
        ite_spearman_correlation_prob = 0.0

    return {
        'true_ate_prob': true_ate_prob,
        'estimated_ate_prob': estimated_ate_prob,
        'ate_bias_prob': ate_bias_prob,
        'ate_rmse_prob': ate_rmse_prob,
        'ite_correlation_prob': ite_correlation_prob,
        'ite_spearman_correlation_prob': ite_spearman_correlation_prob,
    }
