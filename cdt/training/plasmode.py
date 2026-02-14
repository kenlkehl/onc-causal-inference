# cdt/training/plasmode.py
"""Plasmode simulation experiments for sensitivity analysis - CNN-based approach."""

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
from ..data import ClinicalTextDataset, collate_batch
from ..utils import cuda_cleanup, get_memory_info, set_seed


logger = logging.getLogger(__name__)


def run_plasmode_experiments(
    dataset: pd.DataFrame,
    applied_config: AppliedInferenceConfig,
    plasmode_config: PlasmodeExperimentConfig,
    output_path: Path,
    device: torch.device,
    cache=None,  # Kept for API compatibility
    pretrained_weights_path: Optional[Path] = None,
    num_repeats: int = 3,
    num_workers: int = 1,
    gpu_ids: Optional[List[int]] = None
) -> None:
    """
    Run plasmode sensitivity experiments with CNN backbone.
    """
    logger.info("=" * 80)
    logger.info(f"PLASMODE SENSITIVITY EXPERIMENTS - CNN (Workers: {num_workers})")
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

    try:
        metrics, logs = _run_single_plasmode_experiment(
            train_df=task['train_df'],
            scenario=scenario,
            applied_config=task['applied_config'],
            plasmode_config=plasmode_config,
            device=task['device'],
            hyperparams=hyperparams,
            save_dataset_path=save_dataset_path,
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
        cuda_cleanup()


def _run_single_plasmode_experiment(
    train_df: pd.DataFrame,
    scenario: PlasmodeConfig,
    applied_config: AppliedInferenceConfig,
    plasmode_config: PlasmodeExperimentConfig,
    device: torch.device,
    hyperparams: Optional[Dict[str, Any]] = None,
    save_dataset_path: Optional[Path] = None,
) -> Tuple[dict, List[Dict[str, Any]]]:
    """Run a single plasmode experiment."""

    train_fraction = getattr(plasmode_config, 'train_fraction', 0.8)
    seed = hyperparams.get('seed', 42) if hyperparams else 42

    # Split data
    train_split_df, eval_split_df = train_test_split(
        train_df, train_size=train_fraction, random_state=seed
    )
    train_split_df = train_split_df.reset_index(drop=True)
    eval_split_df = eval_split_df.reset_index(drop=True)

    logger.info(f"Single-split: Training on {len(train_split_df)}, evaluating on {len(eval_split_df)}")

    # Step 1: Train generator
    generator, gen_history = _train_cnn_model(
        train_split_df,
        eval_split_df,
        applied_config,
        plasmode_config.generator_architecture,
        plasmode_config.generator_training,
        device
    )

    for entry in gen_history:
        entry['model_type'] = 'generator'
        entry['generation_mode'] = scenario.generation_mode
        entry['scenario_idx'] = hyperparams.get('scenario_idx', -1)
        entry['repeat_idx'] = hyperparams.get('repeat_idx', -1)

    # Step 2: Generate synthetic outcomes
    train_plasmode_df = _generate_plasmode_data(
        train_split_df, generator, scenario, applied_config, device
    )
    eval_plasmode_df = _generate_plasmode_data(
        eval_split_df, generator, scenario, applied_config, device
    )

    train_plasmode_df['sim_split'] = 'train'
    eval_plasmode_df['sim_split'] = 'eval'

    # Step 3: Train evaluator on simulated data
    evaluator, eval_history = _train_cnn_model(
        train_plasmode_df,
        eval_plasmode_df,
        applied_config,
        plasmode_config.evaluator_architecture,
        plasmode_config.evaluator_training,
        device
    )

    for entry in eval_history:
        entry['model_type'] = 'evaluator'
        entry['generation_mode'] = scenario.generation_mode
        entry['scenario_idx'] = hyperparams.get('scenario_idx', -1)
        entry['repeat_idx'] = hyperparams.get('repeat_idx', -1)

    combined_history = gen_history + eval_history

    # Step 4: Generate predictions for eval split
    preds_dict = _predict_cnn_model(evaluator, eval_plasmode_df, applied_config, device)

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
    device: torch.device
) -> Tuple[Union[CausalText, CausalTextForest], List[Dict[str, Any]]]:
    """Train a model with CNN, BERT, or Causal Forest.

    Dispatches to _train_causal_forest_model for model_type="causal_forest".
    """
    # Check for causal forest model type
    model_type = getattr(arch_config, 'model_type', 'dragonnet')
    if model_type == "causal_forest":
        return _train_causal_forest_model(
            train_df, val_df, applied_config, arch_config, train_config, device
        )

    # Get feature extractor type (default to "cnn" for backward compatibility)
    # Normalize type (e.g., "modernbert" -> "bert")
    feature_extractor_type = normalize_feature_extractor_type(
        getattr(arch_config, 'feature_extractor_type', 'cnn')
    )

    model = CausalText(
        feature_extractor_type=feature_extractor_type,
        # CNN args
        embedding_dim=arch_config.cnn_embedding_dim,
        kernel_sizes=arch_config.cnn_kernel_sizes,
        explicit_filter_concepts=arch_config.cnn_explicit_filter_concepts,
        num_kmeans_filters=arch_config.cnn_num_kmeans_filters,
        num_random_filters=arch_config.cnn_num_random_filters,
        cnn_dropout=arch_config.cnn_dropout,
        max_length=arch_config.cnn_max_length,
        min_word_freq=getattr(arch_config, 'cnn_min_word_freq', 2),
        max_vocab_size=getattr(arch_config, 'cnn_max_vocab_size', 50000),
        projection_dim=arch_config.causal_head_representation_dim,
        # BERT args
        bert_model_name=getattr(arch_config, 'bert_model_name', 'bert-base-uncased'),
        bert_max_length=getattr(arch_config, 'bert_max_length', 512),
        bert_projection_dim=getattr(arch_config, 'bert_projection_dim', 128),
        bert_dropout=getattr(arch_config, 'bert_dropout', 0.1),
        bert_freeze_encoder=getattr(arch_config, 'bert_freeze_encoder', False),
        bert_gradient_checkpointing=getattr(arch_config, 'bert_gradient_checkpointing', False),
        # Hierarchical Transformer args
        hier_transformer_sentence_model=getattr(arch_config, 'hier_transformer_sentence_model', 'prajjwal1/bert-tiny'),
        hier_transformer_freeze_sentence_encoder=getattr(arch_config, 'hier_transformer_freeze_sentence_encoder', True),
        hier_transformer_max_chunks=getattr(arch_config, 'hier_transformer_max_chunks', 100),
        hier_transformer_chunk_size=getattr(arch_config, 'hier_transformer_chunk_size', 128),
        hier_transformer_chunk_overlap=getattr(arch_config, 'hier_transformer_chunk_overlap', 32),
        hier_transformer_num_layers=getattr(arch_config, 'hier_transformer_num_layers', 2),
        hier_transformer_num_heads=getattr(arch_config, 'hier_transformer_num_heads', 4),
        hier_transformer_dim=getattr(arch_config, 'hier_transformer_dim', 256),
        hier_transformer_dropout=getattr(arch_config, 'hier_transformer_dropout', 0.1),
        hier_transformer_projection_dim=getattr(arch_config, 'hier_transformer_projection_dim', 128),
        # BERT Cross-Chunk args
        bcc_sentence_model=getattr(arch_config, 'bcc_sentence_model', 'prajjwal1/bert-tiny'),
        bcc_freeze_sentence_encoder=getattr(arch_config, 'bcc_freeze_sentence_encoder', False),
        bcc_max_chunks=getattr(arch_config, 'bcc_max_chunks', 100),
        bcc_chunk_size=getattr(arch_config, 'bcc_chunk_size', 128),
        bcc_chunk_overlap=getattr(arch_config, 'bcc_chunk_overlap', 32),
        bcc_num_cross_layers=getattr(arch_config, 'bcc_num_cross_layers', 2),
        bcc_num_attention_heads=getattr(arch_config, 'bcc_num_attention_heads', 4),
        bcc_cross_chunk_dim=getattr(arch_config, 'bcc_cross_chunk_dim', 256),
        bcc_cross_chunk_dropout=getattr(arch_config, 'bcc_cross_chunk_dropout', 0.1),
        bcc_gated_attention_dim=getattr(arch_config, 'bcc_gated_attention_dim', 128),
        bcc_projection_dim=getattr(arch_config, 'bcc_projection_dim', 128),
        # Gated MIL Hierarchical args
        gated_mil_sentence_model=getattr(arch_config, 'gated_mil_sentence_model', 'prajjwal1/bert-tiny'),
        gated_mil_freeze_sentence_encoder=getattr(arch_config, 'gated_mil_freeze_sentence_encoder', True),
        gated_mil_max_chunks=getattr(arch_config, 'gated_mil_max_chunks', 100),
        gated_mil_chunk_size=getattr(arch_config, 'gated_mil_chunk_size', 128),
        gated_mil_chunk_overlap=getattr(arch_config, 'gated_mil_chunk_overlap', 32),
        gated_mil_hidden_dim=getattr(arch_config, 'gated_mil_hidden_dim', 128),
        gated_mil_num_confounders=getattr(arch_config, 'gated_mil_num_confounders', 4),
        gated_mil_dropout=getattr(arch_config, 'gated_mil_dropout', 0.1),
        gated_mil_projection_dim=getattr(arch_config, 'gated_mil_projection_dim', 128),
        gated_mil_hierarchical=getattr(arch_config, 'gated_mil_hierarchical', False),
        gated_mil_token_hidden_dim=getattr(arch_config, 'gated_mil_token_hidden_dim', 64),
        gated_mil_use_mean_pooling=getattr(arch_config, 'gated_mil_use_mean_pooling', False),
        # GRU-Pool args
        gru_pool_embedding_dim=getattr(arch_config, 'gru_pool_embedding_dim', 128),
        gru_pool_gru_hidden_dim=getattr(arch_config, 'gru_pool_gru_hidden_dim', 128),
        gru_pool_gru_num_layers=getattr(arch_config, 'gru_pool_gru_num_layers', 1),
        gru_pool_gru_bidirectional=getattr(arch_config, 'gru_pool_gru_bidirectional', True),
        gru_pool_gru_dropout=getattr(arch_config, 'gru_pool_gru_dropout', 0.1),
        gru_pool_max_chunks=getattr(arch_config, 'gru_pool_max_chunks', 100),
        gru_pool_chunk_size=getattr(arch_config, 'gru_pool_chunk_size', 128),
        gru_pool_chunk_overlap=getattr(arch_config, 'gru_pool_chunk_overlap', 32),
        gru_pool_transformer_layers=getattr(arch_config, 'gru_pool_transformer_layers', 2),
        gru_pool_transformer_heads=getattr(arch_config, 'gru_pool_transformer_heads', 4),
        gru_pool_transformer_dim=getattr(arch_config, 'gru_pool_transformer_dim', 256),
        gru_pool_gated_attention_dim=getattr(arch_config, 'gru_pool_gated_attention_dim', 128),
        gru_pool_projection_dim=getattr(arch_config, 'gru_pool_projection_dim', 128),
        gru_pool_max_vocab=getattr(arch_config, 'gru_pool_max_vocab', 50000),
        gru_pool_min_word_freq=getattr(arch_config, 'gru_pool_min_word_freq', 2),
        # Conv-Pool args
        conv_pool_embedding_dim=getattr(arch_config, 'conv_pool_embedding_dim', 128),
        conv_pool_conv_dim=getattr(arch_config, 'conv_pool_conv_dim', 256),
        conv_pool_kernel_size=getattr(arch_config, 'conv_pool_kernel_size', 3),
        conv_pool_num_blocks=getattr(arch_config, 'conv_pool_num_blocks', 4),
        conv_pool_dropout=getattr(arch_config, 'conv_pool_dropout', 0.1),
        conv_pool_max_chunks=getattr(arch_config, 'conv_pool_max_chunks', 100),
        conv_pool_chunk_size=getattr(arch_config, 'conv_pool_chunk_size', 128),
        conv_pool_chunk_overlap=getattr(arch_config, 'conv_pool_chunk_overlap', 32),
        conv_pool_transformer_layers=getattr(arch_config, 'conv_pool_transformer_layers', 2),
        conv_pool_transformer_heads=getattr(arch_config, 'conv_pool_transformer_heads', 4),
        conv_pool_transformer_dim=getattr(arch_config, 'conv_pool_transformer_dim', 256),
        conv_pool_transformer_dropout=getattr(arch_config, 'conv_pool_transformer_dropout', 0.1),
        conv_pool_gated_attention_dim=getattr(arch_config, 'conv_pool_gated_attention_dim', 128),
        conv_pool_projection_dim=getattr(arch_config, 'conv_pool_projection_dim', 128),
        conv_pool_max_vocab=getattr(arch_config, 'conv_pool_max_vocab', 50000),
        conv_pool_min_word_freq=getattr(arch_config, 'conv_pool_min_word_freq', 2),
        # Conv1d-Transformer Hybrid args
        c1d_hybrid_embedding_dim=getattr(arch_config, 'c1d_hybrid_embedding_dim', 128),
        c1d_hybrid_conv_dim=getattr(arch_config, 'c1d_hybrid_conv_dim', 256),
        c1d_hybrid_kernel_size=getattr(arch_config, 'c1d_hybrid_kernel_size', 3),
        c1d_hybrid_num_blocks=getattr(arch_config, 'c1d_hybrid_num_blocks', 4),
        c1d_hybrid_conv_dropout=getattr(arch_config, 'c1d_hybrid_conv_dropout', 0.1),
        c1d_hybrid_pool_stride=getattr(arch_config, 'c1d_hybrid_pool_stride', 2),
        c1d_hybrid_max_length=getattr(arch_config, 'c1d_hybrid_max_length', 8192),
        c1d_hybrid_transformer_layers=getattr(arch_config, 'c1d_hybrid_transformer_layers', 2),
        c1d_hybrid_transformer_heads=getattr(arch_config, 'c1d_hybrid_transformer_heads', 4),
        c1d_hybrid_transformer_dim=getattr(arch_config, 'c1d_hybrid_transformer_dim', 256),
        c1d_hybrid_transformer_dropout=getattr(arch_config, 'c1d_hybrid_transformer_dropout', 0.1),
        c1d_hybrid_gated_attention_dim=getattr(arch_config, 'c1d_hybrid_gated_attention_dim', 128),
        c1d_hybrid_projection_dim=getattr(arch_config, 'c1d_hybrid_projection_dim', 128),
        c1d_hybrid_max_vocab=getattr(arch_config, 'c1d_hybrid_max_vocab', 50000),
        c1d_hybrid_min_word_freq=getattr(arch_config, 'c1d_hybrid_min_word_freq', 2),
        # BERT Pool args
        bert_pool_sentence_model=getattr(arch_config, 'bert_pool_sentence_model', 'prajjwal1/bert-tiny'),
        bert_pool_freeze_sentence_encoder=getattr(arch_config, 'bert_pool_freeze_sentence_encoder', False),
        bert_pool_use_pretrained=getattr(arch_config, 'bert_pool_use_pretrained', True),
        bert_pool_max_chunks=getattr(arch_config, 'bert_pool_max_chunks', 100),
        bert_pool_chunk_size=getattr(arch_config, 'bert_pool_chunk_size', 128),
        bert_pool_chunk_overlap=getattr(arch_config, 'bert_pool_chunk_overlap', 32),
        bert_pool_transformer_layers=getattr(arch_config, 'bert_pool_transformer_layers', 2),
        bert_pool_transformer_heads=getattr(arch_config, 'bert_pool_transformer_heads', 4),
        bert_pool_transformer_dim=getattr(arch_config, 'bert_pool_transformer_dim', 256),
        bert_pool_transformer_dropout=getattr(arch_config, 'bert_pool_transformer_dropout', 0.1),
        bert_pool_gated_attention_dim=getattr(arch_config, 'bert_pool_gated_attention_dim', 128),
        bert_pool_projection_dim=getattr(arch_config, 'bert_pool_projection_dim', 128),
        # CLAM instance-level loss args
        clam_enabled=getattr(arch_config, 'clam_enabled', False),
        clam_num_instances=getattr(arch_config, 'clam_num_instances', 5),
        clam_instance_hidden_dim=getattr(arch_config, 'clam_instance_hidden_dim', 64),
        # Contrastive learning args
        contrastive_enabled=getattr(arch_config, 'contrastive_enabled', False),
        contrastive_num_clusters=getattr(arch_config, 'contrastive_num_clusters', 4),
        contrastive_temperature=getattr(arch_config, 'contrastive_temperature', 0.1),
        contrastive_label_mode=getattr(arch_config, 'contrastive_label_mode', 'joint'),
        contrastive_projection_dim=getattr(arch_config, 'contrastive_projection_dim', 64),
        contrastive_min_cluster_size=getattr(arch_config, 'contrastive_min_cluster_size', 2),
        contrastive_clustering_method=getattr(arch_config, 'contrastive_clustering_method', 'kmeans'),
        # LLM args
        llm_model_name=getattr(arch_config, 'llm_model_name', 'Qwen/Qwen3-0.6B-Base'),
        llm_max_length=getattr(arch_config, 'llm_max_length', 8192),
        llm_projection_dim=getattr(arch_config, 'llm_projection_dim', 128),
        llm_dropout=getattr(arch_config, 'llm_dropout', 0.1),
        llm_gradient_checkpointing=getattr(arch_config, 'llm_gradient_checkpointing', True),
        llm_use_pretrained=getattr(arch_config, 'llm_use_pretrained', False),
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
        # Uplift dual extractor mode
        uplift_dual_extractors=getattr(arch_config, 'uplift_dual_extractors', False),
        # DR-MoCE args
        dr_moce_num_experts=getattr(arch_config, 'dr_moce_num_experts', 8),
        dr_moce_router_temperature=getattr(arch_config, 'dr_moce_router_temperature', 1.0),
        dr_moce_propensity_clip=getattr(arch_config, 'dr_moce_propensity_clip', 0.01),
        dr_moce_het_weight=getattr(arch_config, 'dr_moce_het_weight', 0.1),
        dr_moce_balance_weight=getattr(arch_config, 'dr_moce_balance_weight', 0.01),
        dr_moce_crossfit_buffer_size=getattr(arch_config, 'dr_moce_crossfit_buffer_size', 1024),
    )

    train_texts = train_df[applied_config.text_column].tolist()

    if feature_extractor_type == "cnn":
        # CNN-specific initialization
        # Fit tokenizer
        model.fit_tokenizer(train_texts)

        # Initialize embeddings from BERT if configured (unless random init is explicitly requested)
        use_random_init = getattr(arch_config, 'cnn_use_random_embedding_init', False)
        if not use_random_init and getattr(arch_config, 'cnn_init_embeddings_from', None):
            model.feature_extractor.init_embeddings_from_bert(
                arch_config.cnn_init_embeddings_from,
                freeze=getattr(arch_config, 'cnn_freeze_embeddings', False)
            )
        elif use_random_init:
            logger.info("Using random embedding initialization (cnn_use_random_embedding_init=True)")

        # Initialize filters from explicit concepts and/or k-means
        if arch_config.cnn_explicit_filter_concepts or arch_config.cnn_num_kmeans_filters > 0:
            model.feature_extractor.init_filters(
                texts=train_texts,
                freeze=arch_config.cnn_freeze_filters
            )
    elif feature_extractor_type == "gru":
        # GRU-specific initialization
        model.fit_tokenizer(train_texts)
        logger.info(f"Fitted word tokenizer on {len(train_texts)} training texts")

        # Initialize embeddings from BERT if configured
        if getattr(arch_config, 'gru_init_embeddings_from', None):
            model.feature_extractor.init_embeddings_from_bert(
                arch_config.gru_init_embeddings_from,
                freeze=getattr(arch_config, 'gru_freeze_embeddings', False)
            )
    elif feature_extractor_type == "confounder":
        # Confounder extractor initialization
        # Check if GRU-based (requires fit_tokenizer)
        if getattr(arch_config, 'confounder_use_gru', False):
            model.fit_tokenizer(train_texts)
            logger.info(f"Fitted word tokenizer for GRU confounder extractor on {len(train_texts)} texts")
        else:
            # BERT-based or sentence-level: trigger lazy initialization
            model.fit_tokenizer(train_texts)  # No-op for pretrained encoders, triggers init
            logger.info("Using confounder feature extractor (pretrained encoder)")
    elif feature_extractor_type == "hierarchical_transformer":
        # Hierarchical Transformer: trigger lazy initialization
        model.fit_tokenizer(train_texts)  # No-op, triggers init
        logger.info(f"Using Hierarchical Transformer feature extractor: {arch_config.hier_transformer_sentence_model}")
    elif feature_extractor_type == "gated_mil_hierarchical":
        # Gated MIL Hierarchical: trigger lazy initialization
        model.fit_tokenizer(train_texts)  # No-op, triggers init
        logger.info(f"Using Gated MIL Hierarchical feature extractor: {getattr(arch_config, 'gated_mil_sentence_model', 'prajjwal1/bert-tiny')}, "
                   f"{getattr(arch_config, 'gated_mil_num_confounders', 4)} confounders")
    elif feature_extractor_type == "gru_transformer_mil":
        # GRU-Transformer-MIL: requires fit_tokenizer
        model.fit_tokenizer(train_texts)
        logger.info(f"Using GRU-Transformer-MIL feature extractor")
    elif feature_extractor_type == "gru_pool":
        # GRU-Pool: requires fit_tokenizer (learns from scratch)
        model.fit_tokenizer(train_texts)
        logger.info(f"Using GRU-Pool feature extractor")
    elif feature_extractor_type == "conv_pool":
        # Conv-Pool: requires fit_tokenizer (learns from scratch)
        model.fit_tokenizer(train_texts)
        logger.info("Using Dilated Conv Pool feature extractor")
        logger.info(f"  Conv dim: {getattr(arch_config, 'conv_pool_conv_dim', 256)}, "
                   f"kernel_size: {getattr(arch_config, 'conv_pool_kernel_size', 3)}, "
                   f"blocks: {getattr(arch_config, 'conv_pool_num_blocks', 4)}, "
                   f"{getattr(arch_config, 'conv_pool_transformer_layers', 2)} transformer layers")
    elif feature_extractor_type == "conv1d_transformer_hybrid":
        # Conv1d-Transformer Hybrid: requires fit_tokenizer (learns from scratch)
        model.fit_tokenizer(train_texts)
        logger.info("Using Conv1d-Transformer Hybrid feature extractor")
        logger.info(f"  Conv dim: {getattr(arch_config, 'c1d_hybrid_conv_dim', 256)}, "
                   f"kernel_size: {getattr(arch_config, 'c1d_hybrid_kernel_size', 3)}, "
                   f"blocks: {getattr(arch_config, 'c1d_hybrid_num_blocks', 4)}, "
                   f"max_length: {getattr(arch_config, 'c1d_hybrid_max_length', 8192)}, "
                   f"{getattr(arch_config, 'c1d_hybrid_transformer_layers', 2)} transformer layers")
    elif feature_extractor_type == "bert_cross_chunk":
        # BERT Cross-Chunk: trigger lazy initialization (uses pretrained tokenizer)
        model.fit_tokenizer(train_texts)  # No-op, triggers init
        logger.info(f"Using BERT Cross-Chunk feature extractor: {getattr(arch_config, 'bcc_sentence_model', 'prajjwal1/bert-tiny')}")
    elif feature_extractor_type == "llm":
        # LLM uses pretrained tokenizer, no fit_tokenizer needed
        init_mode = "pretrained" if getattr(arch_config, 'llm_use_pretrained', False) else "random init"
        logger.info(f"Using LLM feature extractor: {getattr(arch_config, 'llm_model_name', 'Qwen/Qwen3-0.6B-Base')} ({init_mode})")
    else:
        # BERT uses pretrained tokenizer, no fit_tokenizer needed
        logger.info(f"Using BERT feature extractor: {arch_config.bert_model_name}")

    # Create datasets
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        collate_fn=collate_batch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        collate_fn=collate_batch
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
    clam_instance_weight = getattr(train_config, 'clam_instance_weight', 0.5)
    contrastive_weight = getattr(train_config, 'contrastive_weight', 0.1)

    for epoch in range(train_config.epochs):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            batch['outcome'] = batch['outcome'].to(device)
            batch['treatment'] = batch['treatment'].to(device)

            optimizer.zero_grad()
            losses = model.train_step(
                batch,
                alpha_propensity=train_config.alpha_propensity,
                beta_targreg=train_config.beta_targreg,
                gamma_rlearner=gamma_rlearner,
                gamma_dr=gamma_dr,
                stop_grad_propensity=stop_grad_propensity,
                attention_entropy_weight=attention_entropy_weight,
                clam_instance_weight=clam_instance_weight,
                contrastive_weight=contrastive_weight
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
                losses = model.train_step(
                    batch,
                    alpha_propensity=train_config.alpha_propensity,
                    beta_targreg=train_config.beta_targreg,
                    gamma_rlearner=gamma_rlearner,
                    gamma_dr=gamma_dr,
                    stop_grad_propensity=stop_grad_propensity,
                    attention_entropy_weight=attention_entropy_weight,
                    clam_instance_weight=clam_instance_weight,
                    contrastive_weight=contrastive_weight
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


def _train_causal_forest_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    applied_config: AppliedInferenceConfig,
    arch_config,
    train_config,
    device: torch.device
) -> Tuple[CausalTextForest, List[Dict[str, Any]]]:
    """Train a CausalTextForest model for plasmode experiments.

    Two-stage approach:
        1. Train neural feature extractor with propensity + outcome losses
           (optionally with R-learner loss for representation training)
        2. Train causal forest on extracted features
    """
    feature_extractor_type = normalize_feature_extractor_type(
        getattr(arch_config, 'feature_extractor_type', 'gru_pool')
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
        # GRU-Pool args (most common for causal forest)
        gru_pool_embedding_dim=getattr(arch_config, 'gru_pool_embedding_dim', 128),
        gru_pool_gru_hidden_dim=getattr(arch_config, 'gru_pool_gru_hidden_dim', 128),
        gru_pool_gru_num_layers=getattr(arch_config, 'gru_pool_gru_num_layers', 1),
        gru_pool_gru_bidirectional=getattr(arch_config, 'gru_pool_gru_bidirectional', True),
        gru_pool_gru_dropout=getattr(arch_config, 'gru_pool_gru_dropout', 0.1),
        gru_pool_max_chunks=getattr(arch_config, 'gru_pool_max_chunks', 100),
        gru_pool_chunk_size=getattr(arch_config, 'gru_pool_chunk_size', 128),
        gru_pool_chunk_overlap=getattr(arch_config, 'gru_pool_chunk_overlap', 32),
        gru_pool_transformer_layers=getattr(arch_config, 'gru_pool_transformer_layers', 2),
        gru_pool_transformer_heads=getattr(arch_config, 'gru_pool_transformer_heads', 4),
        gru_pool_transformer_dim=getattr(arch_config, 'gru_pool_transformer_dim', 256),
        gru_pool_gated_attention_dim=getattr(arch_config, 'gru_pool_gated_attention_dim', 128),
        gru_pool_projection_dim=getattr(arch_config, 'gru_pool_projection_dim', 128),
        gru_pool_max_vocab=getattr(arch_config, 'gru_pool_max_vocab', 50000),
        gru_pool_min_word_freq=getattr(arch_config, 'gru_pool_min_word_freq', 2),
        # Conv-Pool args
        conv_pool_embedding_dim=getattr(arch_config, 'conv_pool_embedding_dim', 128),
        conv_pool_conv_dim=getattr(arch_config, 'conv_pool_conv_dim', 256),
        conv_pool_kernel_size=getattr(arch_config, 'conv_pool_kernel_size', 3),
        conv_pool_num_blocks=getattr(arch_config, 'conv_pool_num_blocks', 4),
        conv_pool_dropout=getattr(arch_config, 'conv_pool_dropout', 0.1),
        conv_pool_max_chunks=getattr(arch_config, 'conv_pool_max_chunks', 100),
        conv_pool_chunk_size=getattr(arch_config, 'conv_pool_chunk_size', 128),
        conv_pool_chunk_overlap=getattr(arch_config, 'conv_pool_chunk_overlap', 32),
        conv_pool_transformer_layers=getattr(arch_config, 'conv_pool_transformer_layers', 2),
        conv_pool_transformer_heads=getattr(arch_config, 'conv_pool_transformer_heads', 4),
        conv_pool_transformer_dim=getattr(arch_config, 'conv_pool_transformer_dim', 256),
        conv_pool_transformer_dropout=getattr(arch_config, 'conv_pool_transformer_dropout', 0.1),
        conv_pool_gated_attention_dim=getattr(arch_config, 'conv_pool_gated_attention_dim', 128),
        conv_pool_projection_dim=getattr(arch_config, 'conv_pool_projection_dim', 128),
        conv_pool_max_vocab=getattr(arch_config, 'conv_pool_max_vocab', 50000),
        conv_pool_min_word_freq=getattr(arch_config, 'conv_pool_min_word_freq', 2),
        # Conv1d-Transformer Hybrid args
        c1d_hybrid_embedding_dim=getattr(arch_config, 'c1d_hybrid_embedding_dim', 128),
        c1d_hybrid_conv_dim=getattr(arch_config, 'c1d_hybrid_conv_dim', 256),
        c1d_hybrid_kernel_size=getattr(arch_config, 'c1d_hybrid_kernel_size', 3),
        c1d_hybrid_num_blocks=getattr(arch_config, 'c1d_hybrid_num_blocks', 4),
        c1d_hybrid_conv_dropout=getattr(arch_config, 'c1d_hybrid_conv_dropout', 0.1),
        c1d_hybrid_pool_stride=getattr(arch_config, 'c1d_hybrid_pool_stride', 2),
        c1d_hybrid_max_length=getattr(arch_config, 'c1d_hybrid_max_length', 8192),
        c1d_hybrid_transformer_layers=getattr(arch_config, 'c1d_hybrid_transformer_layers', 2),
        c1d_hybrid_transformer_heads=getattr(arch_config, 'c1d_hybrid_transformer_heads', 4),
        c1d_hybrid_transformer_dim=getattr(arch_config, 'c1d_hybrid_transformer_dim', 256),
        c1d_hybrid_transformer_dropout=getattr(arch_config, 'c1d_hybrid_transformer_dropout', 0.1),
        c1d_hybrid_gated_attention_dim=getattr(arch_config, 'c1d_hybrid_gated_attention_dim', 128),
        c1d_hybrid_projection_dim=getattr(arch_config, 'c1d_hybrid_projection_dim', 128),
        c1d_hybrid_max_vocab=getattr(arch_config, 'c1d_hybrid_max_vocab', 50000),
        c1d_hybrid_min_word_freq=getattr(arch_config, 'c1d_hybrid_min_word_freq', 2),
        # BERT Pool args
        bert_pool_sentence_model=getattr(arch_config, 'bert_pool_sentence_model', 'prajjwal1/bert-tiny'),
        bert_pool_freeze_sentence_encoder=getattr(arch_config, 'bert_pool_freeze_sentence_encoder', False),
        bert_pool_use_pretrained=getattr(arch_config, 'bert_pool_use_pretrained', True),
        bert_pool_max_chunks=getattr(arch_config, 'bert_pool_max_chunks', 100),
        bert_pool_chunk_size=getattr(arch_config, 'bert_pool_chunk_size', 128),
        bert_pool_chunk_overlap=getattr(arch_config, 'bert_pool_chunk_overlap', 32),
        bert_pool_transformer_layers=getattr(arch_config, 'bert_pool_transformer_layers', 2),
        bert_pool_transformer_heads=getattr(arch_config, 'bert_pool_transformer_heads', 4),
        bert_pool_transformer_dim=getattr(arch_config, 'bert_pool_transformer_dim', 256),
        bert_pool_transformer_dropout=getattr(arch_config, 'bert_pool_transformer_dropout', 0.1),
        bert_pool_gated_attention_dim=getattr(arch_config, 'bert_pool_gated_attention_dim', 128),
        bert_pool_projection_dim=getattr(arch_config, 'bert_pool_projection_dim', 128),
        # BERT args (if using BERT extractor)
        bert_model_name=getattr(arch_config, 'bert_model_name', 'bert-base-uncased'),
        bert_max_length=getattr(arch_config, 'bert_max_length', 512),
        bert_projection_dim=getattr(arch_config, 'bert_projection_dim', 128),
        bert_dropout=getattr(arch_config, 'bert_dropout', 0.1),
        bert_freeze_encoder=getattr(arch_config, 'bert_freeze_encoder', False),
        bert_gradient_checkpointing=getattr(arch_config, 'bert_gradient_checkpointing', False),
        # Hierarchical Transformer args
        hier_transformer_sentence_model=getattr(arch_config, 'hier_transformer_sentence_model', 'prajjwal1/bert-tiny'),
        hier_transformer_freeze_sentence_encoder=getattr(arch_config, 'hier_transformer_freeze_sentence_encoder', True),
        hier_transformer_max_chunks=getattr(arch_config, 'hier_transformer_max_chunks', 100),
        hier_transformer_chunk_size=getattr(arch_config, 'hier_transformer_chunk_size', 128),
        hier_transformer_chunk_overlap=getattr(arch_config, 'hier_transformer_chunk_overlap', 32),
        hier_transformer_num_layers=getattr(arch_config, 'hier_transformer_num_layers', 2),
        hier_transformer_num_heads=getattr(arch_config, 'hier_transformer_num_heads', 4),
        hier_transformer_dim=getattr(arch_config, 'hier_transformer_dim', 256),
        hier_transformer_dropout=getattr(arch_config, 'hier_transformer_dropout', 0.1),
        hier_transformer_projection_dim=getattr(arch_config, 'hier_transformer_projection_dim', 128),
        # BERT Cross-Chunk args
        bcc_sentence_model=getattr(arch_config, 'bcc_sentence_model', 'prajjwal1/bert-tiny'),
        bcc_freeze_sentence_encoder=getattr(arch_config, 'bcc_freeze_sentence_encoder', False),
        bcc_max_chunks=getattr(arch_config, 'bcc_max_chunks', 100),
        bcc_chunk_size=getattr(arch_config, 'bcc_chunk_size', 128),
        bcc_chunk_overlap=getattr(arch_config, 'bcc_chunk_overlap', 32),
        bcc_num_cross_layers=getattr(arch_config, 'bcc_num_cross_layers', 2),
        bcc_num_attention_heads=getattr(arch_config, 'bcc_num_attention_heads', 4),
        bcc_cross_chunk_dim=getattr(arch_config, 'bcc_cross_chunk_dim', 256),
        bcc_cross_chunk_dropout=getattr(arch_config, 'bcc_cross_chunk_dropout', 0.1),
        bcc_gated_attention_dim=getattr(arch_config, 'bcc_gated_attention_dim', 128),
        bcc_projection_dim=getattr(arch_config, 'bcc_projection_dim', 128),
        # Gated MIL Hierarchical args
        gated_mil_sentence_model=getattr(arch_config, 'gated_mil_sentence_model', 'prajjwal1/bert-tiny'),
        gated_mil_freeze_sentence_encoder=getattr(arch_config, 'gated_mil_freeze_sentence_encoder', True),
        gated_mil_max_chunks=getattr(arch_config, 'gated_mil_max_chunks', 100),
        gated_mil_chunk_size=getattr(arch_config, 'gated_mil_chunk_size', 128),
        gated_mil_chunk_overlap=getattr(arch_config, 'gated_mil_chunk_overlap', 32),
        gated_mil_hidden_dim=getattr(arch_config, 'gated_mil_hidden_dim', 128),
        gated_mil_num_confounders=getattr(arch_config, 'gated_mil_num_confounders', 4),
        gated_mil_dropout=getattr(arch_config, 'gated_mil_dropout', 0.1),
        gated_mil_projection_dim=getattr(arch_config, 'gated_mil_projection_dim', 128),
        gated_mil_hierarchical=getattr(arch_config, 'gated_mil_hierarchical', False),
        gated_mil_token_hidden_dim=getattr(arch_config, 'gated_mil_token_hidden_dim', 64),
        gated_mil_use_mean_pooling=getattr(arch_config, 'gated_mil_use_mean_pooling', False),
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
        # Contrastive learning args
        contrastive_enabled=getattr(arch_config, 'contrastive_enabled', False),
        contrastive_num_clusters=getattr(arch_config, 'contrastive_num_clusters', 4),
        contrastive_temperature=getattr(arch_config, 'contrastive_temperature', 0.1),
        contrastive_label_mode=getattr(arch_config, 'contrastive_label_mode', 'joint'),
        contrastive_projection_dim=getattr(arch_config, 'contrastive_projection_dim', 64),
        contrastive_min_cluster_size=getattr(arch_config, 'contrastive_min_cluster_size', 2),
        contrastive_clustering_method=getattr(arch_config, 'contrastive_clustering_method', 'kmeans'),
        device=str(device)
    )

    train_texts = train_df[applied_config.text_column].tolist()
    model.fit_tokenizer(train_texts)
    logger.info(f"Using CausalTextForest with {feature_extractor_type.upper()} extractor")

    # Create datasets
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        collate_fn=collate_batch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        collate_fn=collate_batch
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
    contrastive_weight = getattr(train_config, 'contrastive_weight', 0.1)

    # Stage 1: Train representation
    for epoch in range(train_config.epochs):
        model.train()
        epoch_loss = 0.0
        train_r_loss = 0.0

        for batch in train_loader:
            batch['outcome'] = batch['outcome'].to(device)
            batch['treatment'] = batch['treatment'].to(device)

            optimizer.zero_grad()
            losses = model.train_representation_step(
                batch,
                alpha_propensity=alpha_propensity,
                gamma_rlearner=gamma_rlearner,
                label_smoothing=label_smoothing,
                stop_grad_propensity=stop_grad_propensity,
                contrastive_weight=contrastive_weight
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
                losses = model.train_representation_step(
                    batch,
                    alpha_propensity=alpha_propensity,
                    gamma_rlearner=gamma_rlearner,
                    stop_grad_propensity=stop_grad_propensity,
                    contrastive_weight=contrastive_weight
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
    combined_texts = combined_df[applied_config.text_column].tolist()
    combined_T = combined_df[applied_config.treatment_column].values
    combined_Y = combined_df[applied_config.outcome_column].values
    model.train_causal_forest(combined_texts, combined_T, combined_Y)

    return model, history


def _generate_plasmode_data(
    df: pd.DataFrame,
    generator: Union[CausalText, CausalTextForest],
    scenario: PlasmodeConfig,
    applied_config: AppliedInferenceConfig,
    device: torch.device
) -> pd.DataFrame:
    """Generate synthetic outcomes using the generator model."""

    plasmode_df = df.copy()

    # Get features from generator
    dataset = ClinicalTextDataset(
        data=df,
        text_column=applied_config.text_column,
        outcome_column=applied_config.outcome_column,
        treatment_column=applied_config.treatment_column
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_batch
    )

    generator.eval()
    all_features = []

    with torch.no_grad():
        for batch in loader:
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

    # Generate Y0 and Y1 (internally using logit space for proper simulation)
    y0_logit = np.random.randn(len(df)) * scenario.outcome_heterogeneity_scale
    y0_logit += np.log(scenario.baseline_control_outcome_rate / (1 - scenario.baseline_control_outcome_rate))

    y1_logit = y0_logit + ite_logit

    # Sample outcomes
    treatments = df[applied_config.treatment_column].values
    y0_prob = 1 / (1 + np.exp(-y0_logit))
    y1_prob = 1 / (1 + np.exp(-y1_logit))

    observed_prob = np.where(treatments == 1, y1_prob, y0_prob)
    observed_outcome = (np.random.rand(len(df)) < observed_prob).astype(float)

    plasmode_df[applied_config.outcome_column] = observed_outcome
    # Probability scale ground truth only
    plasmode_df['true_y0_prob'] = y0_prob
    plasmode_df['true_y1_prob'] = y1_prob
    plasmode_df['true_ite_prob'] = y1_prob - y0_prob

    return plasmode_df


def _predict_cnn_model(
    model: Union[CausalText, CausalTextForest],
    df: pd.DataFrame,
    applied_config: AppliedInferenceConfig,
    device: torch.device
) -> dict:
    """Generate predictions from CausalText or CausalTextForest model."""

    # Handle CausalTextForest separately (uses different prediction API)
    if isinstance(model, CausalTextForest):
        texts = df[applied_config.text_column].tolist()
        preds = model.predict(texts, return_ci=False)
        return {
            'y0_prob': preds['pred_y0_prob'],
            'y1_prob': preds['pred_y1_prob'],
            'propensity_prob': preds['pred_propensity_prob'],
            'ite_prob': preds['pred_ite_prob']
        }

    # CausalText prediction path
    dataset = ClinicalTextDataset(
        data=df,
        text_column=applied_config.text_column,
        outcome_column=applied_config.outcome_column,
        treatment_column=applied_config.treatment_column
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_batch
    )

    model.eval()
    all_y0 = []
    all_y1 = []
    all_prop = []

    with torch.no_grad():
        for batch in loader:
            texts = batch['texts']
            preds = model.predict(texts)
            all_y0.append(preds['y0_logit'].cpu().numpy())
            all_y1.append(preds['y1_logit'].cpu().numpy())
            all_prop.append(preds['t_logit'].cpu().numpy())

    y0_logit = np.concatenate(all_y0)
    y1_logit = np.concatenate(all_y1)
    prop_logit = np.concatenate(all_prop)
    ite_logit = y1_logit - y0_logit

    # Convert to probabilities using sigmoid
    y0_prob = 1.0 / (1.0 + np.exp(-y0_logit))
    y1_prob = 1.0 / (1.0 + np.exp(-y1_logit))
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
