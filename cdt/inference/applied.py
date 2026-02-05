# cdt/inference/applied.py
"""Applied causal inference on real clinical data with CNN or BERT feature extraction."""

import gc
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

from ..config import AppliedInferenceConfig, normalize_feature_extractor_type, ExplicitConfounderSpec
from ..models.causal_text import CausalText
from ..data import (
    ClinicalTextDataset,
    collate_batch,
    load_dataset
)
from ..utils import cuda_cleanup, get_memory_info
from ..extraction import VLLMConfounderExtractor, ExtractionCache

# Import forest inference (lazy to avoid import errors if econml not installed)
def _get_forest_inference():
    from .applied_forest import run_applied_inference_forest
    return run_applied_inference_forest


logger = logging.getLogger(__name__)


def _get_explicit_confounder_specs(config: AppliedInferenceConfig) -> Optional[List[ExplicitConfounderSpec]]:
    """Get explicit confounder specs from config if enabled."""
    if hasattr(config, 'explicit_confounders') and config.explicit_confounders.enabled:
        return config.explicit_confounders.confounders
    return None


def _run_explicit_confounder_extraction(
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    output_path: Path
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Run LLM-based explicit confounder extraction as a preprocessing step.

    Args:
        dataset: Input DataFrame with clinical text
        config: Configuration with explicit_confounders settings
        output_path: Output path (used for cache location)

    Returns:
        Tuple of (enriched_dataset, confounder_column_names)
    """
    logger.info("=" * 80)
    logger.info("EXPLICIT CONFOUNDER EXTRACTION (LLM)")
    logger.info("=" * 80)

    conf_config = config.explicit_confounders
    specs = conf_config.confounders

    logger.info(f"Extracting {len(specs)} confounders: {[s.name for s in specs]}")
    logger.info(f"vLLM mode: {conf_config.vllm_mode}, model: {conf_config.vllm_model_name}")

    # Check cache
    cache = ExtractionCache(cache_dir=conf_config.cache_dir)
    cache_config = {
        'confounders': specs,
        'vllm_model_name': conf_config.vllm_model_name,
        'extraction_temperature': conf_config.extraction_temperature,
        'extraction_max_tokens': conf_config.extraction_max_tokens,
    }

    cached_df = None
    if conf_config.cache_enabled:
        cached_df = cache.load_if_valid(
            config.dataset_path,
            cache_config,
            expected_rows=len(dataset)
        )

    if cached_df is not None:
        logger.info("Using cached extraction results")
        # Merge cached columns into dataset
        confounder_columns = [f"explicit_conf_{s.name}" for s in specs]
        for col in cached_df.columns:
            dataset[col] = cached_df[col].values
        return dataset, confounder_columns

    # Run extraction
    logger.info(f"Running LLM extraction on {len(dataset)} texts...")
    texts = dataset[config.text_column].tolist()

    extractor = VLLMConfounderExtractor(
        specs=specs,
        mode=conf_config.vllm_mode,
        server_url=conf_config.vllm_server_url or "http://localhost:8000/v1",
        model_name=conf_config.vllm_model_name,
        tensor_parallel_size=conf_config.vllm_tensor_parallel_size,
        gpu_memory_utilization=conf_config.vllm_gpu_memory_utilization,
        download_dir=conf_config.vllm_download_dir,
        max_retries=conf_config.extraction_max_retries,
        temperature=conf_config.extraction_temperature,
        max_tokens=conf_config.extraction_max_tokens
    )

    try:
        extracted_df = extractor.extract_to_dataframe(
            texts,
            batch_size=conf_config.extraction_batch_size
        )
    finally:
        extractor.cleanup()

    # Merge extracted columns into dataset
    confounder_columns = [f"explicit_conf_{s.name}" for s in specs]
    for col in extracted_df.columns:
        dataset[col] = extracted_df[col].values

    # Log extraction statistics
    for spec in specs:
        col = f"explicit_conf_{spec.name}"
        missing_col = f"{col}_missing"
        if missing_col in dataset.columns:
            missing_count = dataset[missing_col].sum()
            logger.info(f"  {spec.name}: {len(dataset) - missing_count}/{len(dataset)} extracted "
                       f"({missing_count} missing)")

    # Save cache
    if conf_config.cache_enabled:
        cache.save(config.dataset_path, cache_config, extracted_df)

    logger.info("=" * 80)
    logger.info("CONTINUING WITH MODEL TRAINING")
    logger.info("=" * 80)

    return dataset, confounder_columns


def run_applied_inference(
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    output_path: Path,
    device: torch.device,
    cache=None,  # Kept for API compatibility, not used
    pretrained_weights_path: Optional[Path] = None,
    gpu_ids: Optional[List[int]] = None,
    num_workers: int = 1,
    save_filter_interpretations: bool = False,
    filter_interpretation_top_k: int = 10,
    save_confounder_interpretations: bool = False,
    confounder_interpretation_top_k: int = 5
) -> None:
    """
    Run applied causal inference on real data using CNN backbone.

    Args:
        dataset: DataFrame with clinical text, outcomes, and treatments
        config: Configuration for applied inference
        output_path: Path to save predictions
        device: PyTorch device
        cache: Unused, kept for API compatibility
        pretrained_weights_path: Unused, kept for API compatibility
        gpu_ids: List of GPU IDs for parallel processing
        num_workers: Number of parallel workers
        save_filter_interpretations: Whether to save filter interpretation analysis
        filter_interpretation_top_k: Number of top n-grams per filter to save
        save_confounder_interpretations: Whether to save confounder attention analysis
        confounder_interpretation_top_k: Number of top-attended sentences per confounder
    """
    # Explicit confounder extraction (if enabled)
    explicit_confounder_columns = None
    if hasattr(config, 'explicit_confounders') and config.explicit_confounders.enabled:
        dataset, explicit_confounder_columns = _run_explicit_confounder_extraction(
            dataset, config, output_path
        )

    # Route to causal forest inference if model_type is "causal_forest"
    if hasattr(config, 'architecture') and config.architecture.model_type == "causal_forest":
        logger.info("Routing to Causal Forest inference pipeline")
        run_forest_inference = _get_forest_inference()
        run_forest_inference(
            dataset=dataset,
            config=config,
            output_path=output_path,
            device=device,
            num_workers=num_workers,
            explicit_confounder_columns=explicit_confounder_columns
        )
        return

    logger.info("=" * 80)
    logger.info("APPLIED CAUSAL INFERENCE (CNN)")
    logger.info("=" * 80)

    # Propensity trimming preprocessing (if enabled)
    trimming_stats = None
    if hasattr(config, 'propensity_trimming') and config.propensity_trimming.enabled:
        logger.info("=" * 80)
        logger.info("PROPENSITY-BASED DATASET TRIMMING")
        logger.info("=" * 80)

        from ..training.propensity_trimming import (
            train_propensity_model_cv, trim_by_propensity
        )

        # Train propensity model with CV to get out-of-sample scores
        dataset, propensity_training_log = train_propensity_model_cv(
            dataset, config, device, num_workers, gpu_ids
        )

        # Save propensity model training log
        training_log_path = output_path.parent / "propensity_trimming_training_log.csv"
        propensity_training_log.to_csv(training_log_path, index=False)
        logger.info(f"Propensity training log saved to: {training_log_path}")

        original_size = len(dataset)

        # Trim dataset
        dataset, trimming_stats = trim_by_propensity(
            dataset,
            config.propensity_trimming.min_propensity,
            config.propensity_trimming.max_propensity
        )

        logger.info(f"Dataset trimmed: {original_size} -> {len(dataset)} "
                   f"({trimming_stats['removed_low']} below min, "
                   f"{trimming_stats['removed_high']} above max)")

        # Save trimming stats
        trimming_stats_path = output_path.parent / "propensity_trimming_stats.json"
        with open(trimming_stats_path, 'w') as f:
            json.dump(trimming_stats, f, indent=2)
        logger.info(f"Trimming stats saved to: {trimming_stats_path}")

        logger.info("=" * 80)
        logger.info("CONTINUING WITH DRAGONNET TRAINING ON TRIMMED DATASET")
        logger.info("=" * 80)

    # Outcome model training (if enabled) - for assessing prognostic signal
    if hasattr(config, 'outcome_model') and config.outcome_model.enabled:
        logger.info("=" * 80)
        logger.info("OUTCOME MODEL TRAINING (PROGNOSTIC SIGNAL ASSESSMENT)")
        logger.info("=" * 80)

        from ..training.outcome_training import train_outcome_model_cv

        # Train outcome model with CV to get out-of-sample scores
        dataset, outcome_training_log = train_outcome_model_cv(
            dataset, config, device, num_workers, gpu_ids
        )

        # Save outcome model training log
        training_log_path = output_path.parent / "outcome_model_training_log.csv"
        outcome_training_log.to_csv(training_log_path, index=False)
        logger.info(f"Outcome model training log saved to: {training_log_path}")

        # Log summary of outcome prediction performance
        if 'val_auroc' in outcome_training_log.columns:
            mean_auroc = outcome_training_log['val_auroc'].dropna().mean()
            logger.info(f"Mean validation AUROC across folds: {mean_auroc:.4f}")

        logger.info("=" * 80)
        logger.info("CONTINUING WITH DRAGONNET TRAINING")
        logger.info("=" * 80)

    # Determine mode
    if config.cv_folds > 1:
        _run_cv_inference(
            dataset, config, output_path, device, gpu_ids, num_workers,
            save_filter_interpretations, filter_interpretation_top_k,
            save_confounder_interpretations, confounder_interpretation_top_k,
            explicit_confounder_columns=explicit_confounder_columns
        )
    else:
        _run_fixed_split_inference(
            dataset, config, output_path, device,
            save_filter_interpretations, filter_interpretation_top_k,
            save_confounder_interpretations, confounder_interpretation_top_k,
            explicit_confounder_columns=explicit_confounder_columns
        )


def _run_cv_inference(
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    output_path: Path,
    device: torch.device,
    gpu_ids: Optional[List[int]] = None,
    num_workers: int = 1,
    save_filter_interpretations: bool = False,
    filter_interpretation_top_k: int = 10,
    save_confounder_interpretations: bool = False,
    confounder_interpretation_top_k: int = 5,
    explicit_confounder_columns: Optional[List[str]] = None
) -> None:
    """Run K-Fold Cross-Validation inference."""
    k = config.cv_folds
    logger.info(f"Starting {k}-Fold Cross-Validation on {len(dataset)} samples")

    # Reset index to ensure KFold works with indices
    dataset = dataset.reset_index(drop=True)

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    splits = list(kf.split(dataset))

    # Determine devices to use
    if gpu_ids and device.type == "cuda":
        devices = [torch.device(f"cuda:{i}") for i in gpu_ids]
    else:
        # MPS and CPU are single-device; ignore gpu_ids
        devices = [device]

    if num_workers > 1:
        logger.info(f"Parallelizing across {num_workers} workers on devices: {devices}")

        results = Parallel(n_jobs=num_workers)(
            delayed(_process_fold)(
                fold, train_idx, test_idx, dataset, config,
                devices[fold % len(devices)],
                explicit_confounder_columns=explicit_confounder_columns
            )
            for fold, (train_idx, test_idx) in enumerate(splits)
        )
    else:
        results = []
        for fold, (train_idx, test_idx) in enumerate(splits):
            results.append(_process_fold(
                fold, train_idx, test_idx, dataset, config,
                devices[0],
                explicit_confounder_columns=explicit_confounder_columns
            ))

    # Unpack results
    all_predictions = [r[0] for r in results]
    all_training_logs = [log for r in results for log in r[1]]

    # Combine predictions and save
    results_df = pd.concat(all_predictions).sort_index()
    _save_and_summarize(results_df, output_path)

    # Save training logs
    log_path = output_path.parent / "training_log.csv"
    pd.DataFrame(all_training_logs).to_csv(log_path, index=False)
    logger.info(f"Training logs saved to: {log_path}")

    # Save filter interpretations for the last fold if requested
    # (In CV mode, we train one more model on the last fold's training data for interpretation)
    if save_filter_interpretations or save_confounder_interpretations:
        logger.info("Generating interpretations from final fold model...")
        last_fold = k - 1
        train_idx, _ = splits[last_fold]
        train_df = dataset.iloc[train_idx]
        val_df = dataset.iloc[splits[last_fold][1]]  # Use test as val for this

        # Train a model on the last fold for interpretation
        model, _ = _train_single_model(train_df, val_df, config, devices[0])
        train_texts = train_df[config.text_column].tolist()

        if save_filter_interpretations:
            _save_filter_interpretations(
                model, train_texts, output_path.parent,
                top_k=filter_interpretation_top_k
            )

        if save_confounder_interpretations:
            _save_confounder_interpretations(
                model, train_texts, output_path.parent,
                top_k=confounder_interpretation_top_k
            )

        # Cleanup
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()


def _process_fold(
    fold: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    device: torch.device,
    explicit_confounder_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Process a single fold (can be run in parallel)."""
    # Re-configure logger for worker process
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    logger.info(f"FOLD {fold + 1} starting on {device}")

    # 1. Prepare Data for this Fold
    train_df = dataset.iloc[train_idx]
    test_df = dataset.iloc[test_idx]

    # 2. Train Model on this fold
    model, history = _train_single_model(
        train_df, test_df, config, device,
        explicit_confounder_columns=explicit_confounder_columns
    )

    # Log History
    for entry in history:
        entry['fold'] = fold + 1

    # 3. Predict on Held-out Test fold
    preds = _predict_dataset(
        model, test_df, config, device,
        explicit_confounder_columns=explicit_confounder_columns
    )

    # 4. Store predictions with indices to reconstruct dataframe
    preds_df = test_df.copy()
    # Probability scale predictions (pred_* prefix indicates predicted values)
    preds_df['pred_y0_prob'] = preds['y0_prob']
    preds_df['pred_y1_prob'] = preds['y1_prob']
    preds_df['pred_ite_prob'] = preds['ite_prob']
    preds_df['pred_propensity_prob'] = preds['propensity_prob']
    preds_df['cv_fold'] = fold + 1

    # Aggressive GPU cleanup to prevent OOM across folds
    model.cpu()
    del model
    del preds
    del train_df, test_df

    gc.collect()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif device.type == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()

    gc.collect()
    cuda_cleanup()

    logger.info(f"FOLD {fold + 1} complete | {get_memory_info()}")
    return preds_df, history


def _run_fixed_split_inference(
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    output_path: Path,
    device: torch.device,
    save_filter_interpretations: bool = False,
    filter_interpretation_top_k: int = 10,
    save_confounder_interpretations: bool = False,
    confounder_interpretation_top_k: int = 5,
    explicit_confounder_columns: Optional[List[str]] = None
) -> None:
    """Run inference using fixed train/val/test splits."""
    logger.info("Running Fixed Split Inference (Train/Val/Test)")

    # Split data
    train_df = dataset[dataset[config.split_column] == 'train'].copy()
    val_df = dataset[dataset[config.split_column] == 'val'].copy()
    test_df = dataset[dataset[config.split_column] == 'test'].copy()

    logger.info(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Train
    model, history = _train_single_model(
        train_df, val_df, config, device,
        explicit_confounder_columns=explicit_confounder_columns
    )

    # Save training logs
    log_path = output_path.parent / "training_log.csv"
    pd.DataFrame(history).to_csv(log_path, index=False)
    logger.info(f"Training logs saved to: {log_path}")

    # Save interpretations if requested
    train_texts = train_df[config.text_column].tolist()

    if save_filter_interpretations:
        _save_filter_interpretations(
            model, train_texts, output_path.parent,
            top_k=filter_interpretation_top_k
        )

    if save_confounder_interpretations:
        _save_confounder_interpretations(
            model, train_texts, output_path.parent,
            top_k=confounder_interpretation_top_k
        )

    # Predict on Test
    logger.info("Generating predictions on test set...")
    preds = _predict_dataset(
        model, test_df, config, device,
        explicit_confounder_columns=explicit_confounder_columns
    )

    # Combine predictions (probability scale, pred_* prefix indicates predicted values)
    results_df = test_df.copy()
    results_df['pred_y0_prob'] = preds['y0_prob']
    results_df['pred_y1_prob'] = preds['y1_prob']
    results_df['pred_ite_prob'] = preds['ite_prob']
    results_df['pred_propensity_prob'] = preds['propensity_prob']

    _save_and_summarize(results_df, output_path)


def _train_single_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: AppliedInferenceConfig,
    device: torch.device,
    explicit_confounder_columns: Optional[List[str]] = None
) -> Tuple[CausalText, List[Dict[str, Any]]]:
    """Train a single model instance (CNN or BERT feature extractor)."""
    arch_config = config.architecture

    # Get feature extractor type (default to "cnn" for backward compatibility)
    # Normalize type (e.g., "modernbert" -> "bert")
    feature_extractor_type = normalize_feature_extractor_type(
        getattr(arch_config, 'feature_extractor_type', 'cnn')
    )

    # Determine max_length and embedding_dim based on extractor type
    if feature_extractor_type == "gru":
        max_length = getattr(arch_config, 'gru_max_length', 8192)
        embedding_dim = getattr(arch_config, 'gru_embedding_dim', 256)
        min_word_freq = getattr(arch_config, 'gru_min_word_freq', 2)
        max_vocab_size = getattr(arch_config, 'gru_max_vocab_size', 50000)
    else:
        max_length = arch_config.cnn_max_length
        embedding_dim = arch_config.cnn_embedding_dim
        min_word_freq = getattr(arch_config, 'cnn_min_word_freq', 2)
        max_vocab_size = getattr(arch_config, 'cnn_max_vocab_size', 50000)

    # Create model with appropriate feature extractor
    model = CausalText(
        feature_extractor_type=feature_extractor_type,
        # CNN/GRU shared args
        embedding_dim=embedding_dim,
        max_length=max_length,
        min_word_freq=min_word_freq,
        max_vocab_size=max_vocab_size,
        # CNN-specific args
        kernel_sizes=arch_config.cnn_kernel_sizes,
        explicit_filter_concepts=arch_config.cnn_explicit_filter_concepts,
        num_kmeans_filters=arch_config.cnn_num_kmeans_filters,
        num_random_filters=arch_config.cnn_num_random_filters,
        cnn_dropout=arch_config.cnn_dropout,
        projection_dim=arch_config.causal_head_representation_dim,
        # BERT args
        bert_model_name=getattr(arch_config, 'bert_model_name', 'bert-base-uncased'),
        bert_max_length=getattr(arch_config, 'bert_max_length', 512),
        bert_projection_dim=getattr(arch_config, 'bert_projection_dim', 128),
        bert_dropout=getattr(arch_config, 'bert_dropout', 0.1),
        bert_freeze_encoder=getattr(arch_config, 'bert_freeze_encoder', False),
        bert_gradient_checkpointing=getattr(arch_config, 'bert_gradient_checkpointing', False),
        # GRU args
        gru_hidden_dim=getattr(arch_config, 'gru_hidden_dim', 256),
        gru_num_layers=getattr(arch_config, 'gru_num_layers', 2),
        gru_dropout=getattr(arch_config, 'gru_dropout', 0.1),
        gru_bidirectional=getattr(arch_config, 'gru_bidirectional', True),
        gru_attention_dim=getattr(arch_config, 'gru_attention_dim', None),
        gru_projection_dim=getattr(arch_config, 'gru_projection_dim', 128),
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
        # CLAM instance-level loss args
        clam_enabled=getattr(arch_config, 'clam_enabled', False),
        clam_num_instances=getattr(arch_config, 'clam_num_instances', 5),
        clam_instance_hidden_dim=getattr(arch_config, 'clam_instance_hidden_dim', 64),
        # LLM args (decoder-only with random init)
        llm_model_name=getattr(arch_config, 'llm_model_name', 'Qwen/Qwen3-0.6B-Base'),
        llm_max_length=getattr(arch_config, 'llm_max_length', 8192),
        llm_projection_dim=getattr(arch_config, 'llm_projection_dim', 128),
        llm_dropout=getattr(arch_config, 'llm_dropout', 0.1),
        llm_gradient_checkpointing=getattr(arch_config, 'llm_gradient_checkpointing', True),
        # Numeric feature args
        numeric_features_enabled=getattr(arch_config, 'numeric_features_enabled', False),
        numeric_embedding_dim=getattr(arch_config, 'numeric_embedding_dim', 32),
        numeric_magnitude_bins=getattr(arch_config, 'numeric_magnitude_bins', 8),
        numeric_type_categories=getattr(arch_config, 'numeric_type_categories', 10),
        # Explicit confounder featurizer args
        explicit_confounder_specs=_get_explicit_confounder_specs(config) if explicit_confounder_columns else None,
        explicit_confounder_output_dim=getattr(config.explicit_confounders, 'featurizer_output_dim', 64) if hasattr(config, 'explicit_confounders') else 64,
        explicit_confounder_hidden_dim=getattr(config.explicit_confounders, 'featurizer_hidden_dim', 128) if hasattr(config, 'explicit_confounders') else 128,
        explicit_confounder_dropout=getattr(config.explicit_confounders, 'featurizer_dropout', 0.1) if hasattr(config, 'explicit_confounders') else 0.1,
        # Causal head args
        causal_head_representation_dim=arch_config.causal_head_representation_dim,
        causal_head_hidden_outcome_dim=arch_config.causal_head_hidden_outcome_dim,
        causal_head_dropout=getattr(arch_config, 'causal_head_dropout', 0.2),
        device=str(device),
        model_type=arch_config.model_type,
        # R-Learner dual extractor mode
        rlearner_dual_extractors=getattr(arch_config, 'rlearner_dual_extractors', False),
        # Uplift dual extractor mode
        uplift_dual_extractors=getattr(arch_config, 'uplift_dual_extractors', False),
    )
    logger.info(f"Created model with {feature_extractor_type.upper()} feature extractor")

    train_texts = train_df[config.text_column].tolist()

    if feature_extractor_type == "cnn":
        # CNN-specific initialization
        # Fit word tokenizer on training texts
        model.fit_tokenizer(train_texts)
        logger.info(f"Fitted word tokenizer on {len(train_texts)} training texts")

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
        # Fit word tokenizer on training texts
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
        logger.info(f"Using GRU-Pool feature extractor: "
                   f"GRU {getattr(arch_config, 'gru_pool_gru_hidden_dim', 128)}x{2 if getattr(arch_config, 'gru_pool_gru_bidirectional', True) else 1}, "
                   f"{getattr(arch_config, 'gru_pool_transformer_layers', 2)} transformer layers")
    elif feature_extractor_type == "llm":
        # LLM uses pretrained tokenizer, no fit_tokenizer needed
        logger.info(f"Using LLM feature extractor: {getattr(arch_config, 'llm_model_name', 'Qwen/Qwen3-0.6B-Base')} (random init)")
    else:
        # BERT uses pretrained tokenizer, no fit_tokenizer needed
        logger.info(f"Using BERT feature extractor: {arch_config.bert_model_name}")

    # Fit explicit confounder featurizer if specs provided
    if explicit_confounder_columns and model.explicit_confounder_featurizer is not None:
        # Extract confounder values from training data for fitting normalization stats
        train_confounder_values = []
        for idx in range(len(train_df)):
            row_values = {}
            for col in explicit_confounder_columns:
                row_values[col] = train_df[col].iloc[idx]
                missing_col = f"{col}_missing"
                if missing_col in train_df.columns:
                    row_values[f"{col}_missing"] = train_df[missing_col].iloc[idx]
            train_confounder_values.append(row_values)
        model.fit_explicit_confounder_featurizer(train_confounder_values)
        logger.info(f"Fitted explicit confounder featurizer on {len(train_confounder_values)} training samples")

    # Create datasets
    train_dataset = ClinicalTextDataset(
        data=train_df,
        text_column=config.text_column,
        outcome_column=config.outcome_column,
        treatment_column=config.treatment_column,
        explicit_confounder_columns=explicit_confounder_columns
    )

    val_dataset = ClinicalTextDataset(
        data=val_df,
        text_column=config.text_column,
        outcome_column=config.outcome_column,
        treatment_column=config.treatment_column,
        explicit_confounder_columns=explicit_confounder_columns
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate_batch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_batch
    )

    # Optimization
    train_config = config.training
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=getattr(train_config, 'weight_decay', 0.01)
    )

    if train_config.lr_schedule == "linear":
        total_steps = len(train_loader) * train_config.epochs
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps
        )
    else:
        scheduler = None

    # Training Loop
    best_val_loss = float('inf')
    best_model_state = None
    history = []

    for epoch in range(train_config.epochs):
        model.train()
        train_stats = _train_epoch(model, train_loader, optimizer, scheduler, device, train_config)

        model.eval()
        val_stats = _eval_epoch(model, val_loader, device, train_config)

        # Record history
        epoch_log = {
            'epoch': epoch + 1,
            'train_loss': train_stats['loss'],
            'train_auroc_y0': train_stats['auroc_y0'],
            'train_auroc_y1': train_stats['auroc_y1'],
            'train_auroc_prop': train_stats['auroc_prop'],
            'val_loss': val_stats['loss'],
            'val_auroc_y0': val_stats['auroc_y0'],
            'val_auroc_y1': val_stats['auroc_y1'],
            'val_auroc_prop': val_stats['auroc_prop'],
        }
        history.append(epoch_log)

        # Save best
        if val_stats['loss'] < best_val_loss:
            best_val_loss = val_stats['loss']
            best_model_state = model.state_dict()

    # Restore best
    if best_model_state:
        model.load_state_dict(best_model_state)

    # Cleanup training artifacts
    del train_loader, val_loader, train_dataset, val_dataset
    del optimizer, best_model_state
    if scheduler is not None:
        del scheduler
    gc.collect()

    return model, history


def _predict_dataset(
    model: CausalText,
    df: pd.DataFrame,
    config: AppliedInferenceConfig,
    device: torch.device,
    explicit_confounder_columns: Optional[List[str]] = None
) -> dict:
    """Generate predictions for a dataframe."""
    dataset = ClinicalTextDataset(
        data=df,
        text_column=config.text_column,
        outcome_column=config.outcome_column,
        treatment_column=config.treatment_column,
        explicit_confounder_columns=explicit_confounder_columns
    )

    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_batch
    )

    result = _generate_predictions(model, loader, device)

    del loader, dataset
    gc.collect()

    return result


def _train_epoch(
    model: CausalText,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    config
) -> dict:
    """Train for one epoch."""
    epoch_loss = 0.0
    all_targets = []
    all_treatments = []
    all_y0 = []
    all_y1 = []
    all_prop = []

    # Get regularization options from config
    label_smoothing = getattr(config, 'label_smoothing', 0.0)
    gradient_clip_norm = getattr(config, 'gradient_clip_norm', 0.0)
    gamma_rlearner = getattr(config, 'gamma_rlearner', 1.0)
    stop_grad_propensity = getattr(config, 'stop_grad_propensity', False)
    attention_entropy_weight = getattr(config, 'attention_entropy_weight', 0.0)
    clam_instance_weight = getattr(config, 'clam_instance_weight', 0.5)

    for batch in tqdm(loader, desc="Training", leave=False):
        # Move tensors to device
        batch['outcome'] = batch['outcome'].to(device)
        batch['treatment'] = batch['treatment'].to(device)
        # 'texts' stays as list of strings

        optimizer.zero_grad()

        losses = model.train_step(
            batch,
            alpha_propensity=config.alpha_propensity,
            beta_targreg=config.beta_targreg,
            gamma_rlearner=gamma_rlearner,
            label_smoothing=label_smoothing,
            stop_grad_propensity=stop_grad_propensity,
            attention_entropy_weight=attention_entropy_weight,
            clam_instance_weight=clam_instance_weight
        )

        losses['loss'].backward()

        # Gradient clipping (if enabled)
        if gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        epoch_loss += losses['loss'].item()

        # Collect for metrics
        all_targets.append(batch['outcome'].detach().cpu())
        all_treatments.append(batch['treatment'].detach().cpu())
        all_y0.append(losses['y0_logit'].detach().cpu())
        all_y1.append(losses['y1_logit'].detach().cpu())
        all_prop.append(losses['t_logit'].detach().cpu())

    return _compute_epoch_metrics(epoch_loss, loader, all_targets, all_treatments, all_y0, all_y1, all_prop)


def _eval_epoch(
    model: CausalText,
    loader: DataLoader,
    device: torch.device,
    config
) -> dict:
    """Evaluate for one epoch."""
    epoch_loss = 0.0
    all_targets = []
    all_treatments = []
    all_y0 = []
    all_y1 = []
    all_prop = []

    gamma_rlearner = getattr(config, 'gamma_rlearner', 1.0)
    stop_grad_propensity = getattr(config, 'stop_grad_propensity', False)
    attention_entropy_weight = getattr(config, 'attention_entropy_weight', 0.0)
    clam_instance_weight = getattr(config, 'clam_instance_weight', 0.5)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            batch['outcome'] = batch['outcome'].to(device)
            batch['treatment'] = batch['treatment'].to(device)

            losses = model.train_step(
                batch,
                alpha_propensity=config.alpha_propensity,
                beta_targreg=config.beta_targreg,
                gamma_rlearner=gamma_rlearner,
                stop_grad_propensity=stop_grad_propensity,
                attention_entropy_weight=attention_entropy_weight,
                clam_instance_weight=clam_instance_weight
            )

            epoch_loss += losses['loss'].item()

            all_targets.append(batch['outcome'].detach().cpu())
            all_treatments.append(batch['treatment'].detach().cpu())
            all_y0.append(losses['y0_logit'].detach().cpu())
            all_y1.append(losses['y1_logit'].detach().cpu())
            all_prop.append(losses['t_logit'].detach().cpu())

    return _compute_epoch_metrics(epoch_loss, loader, all_targets, all_treatments, all_y0, all_y1, all_prop)


def _compute_epoch_metrics(epoch_loss, loader, all_targets, all_treatments, all_y0, all_y1, all_prop):
    """Helper to compute AUROCs from collected batch outputs."""
    y_true = torch.cat(all_targets).numpy()
    t_true = torch.cat(all_treatments).numpy()
    y0_scores = torch.cat(all_y0).numpy()
    y1_scores = torch.cat(all_y1).numpy()
    prop_scores = torch.sigmoid(torch.cat(all_prop)).numpy()

    # Safe AUROC calculation
    def safe_auc(y, score):
        try:
            if len(np.unique(y)) < 2:
                return None
            return roc_auc_score(y, score)
        except Exception:
            return None

    # AUROC Y0 (on T=0 samples)
    mask0 = (t_true == 0)
    auroc_y0 = safe_auc(y_true[mask0], y0_scores[mask0]) if mask0.any() else None

    # AUROC Y1 (on T=1 samples)
    mask1 = (t_true == 1)
    auroc_y1 = safe_auc(y_true[mask1], y1_scores[mask1]) if mask1.any() else None

    # AUROC Propensity
    auroc_prop = safe_auc(t_true, prop_scores)

    return {
        'loss': epoch_loss / len(loader),
        'auroc_y0': auroc_y0,
        'auroc_y1': auroc_y1,
        'auroc_prop': auroc_prop
    }


def _generate_predictions(
    model: CausalText,
    loader: DataLoader,
    device: torch.device
) -> dict:
    """Generate predictions on test set."""
    all_y0 = []
    all_y1 = []
    all_propensity = []

    model.eval()

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting", leave=False):
            texts = batch['texts']
            explicit_confounder_values = batch.get('explicit_confounder_values', None)

            preds = model.predict(texts, explicit_confounder_values=explicit_confounder_values)

            all_y0.append(preds['y0_logit'].cpu().numpy())
            all_y1.append(preds['y1_logit'].cpu().numpy())
            all_propensity.append(preds['t_logit'].cpu().numpy())

    y0_logit = np.concatenate(all_y0)
    y1_logit = np.concatenate(all_y1)
    propensity_logit = np.concatenate(all_propensity)
    ite_logit = y1_logit - y0_logit

    # Convert to probabilities using sigmoid
    y0_prob = 1.0 / (1.0 + np.exp(-y0_logit))
    y1_prob = 1.0 / (1.0 + np.exp(-y1_logit))
    propensity_prob = 1.0 / (1.0 + np.exp(-propensity_logit))
    ite_prob = y1_prob - y0_prob

    return {
        'y0_prob': y0_prob,
        'y1_prob': y1_prob,
        'propensity_prob': propensity_prob,
        'ite_prob': ite_prob
    }


def _save_and_summarize(results_df: pd.DataFrame, output_path: Path) -> None:
    """Save results and print summary."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(output_path, index=False)

    logger.info(f"Predictions saved to: {output_path}")
    logger.info("\nPrediction Summary:")
    logger.info(f"  Samples: {len(results_df)}")
    logger.info("  Predicted ITE (probability scale):")
    logger.info(f"    Mean: {results_df['pred_ite_prob'].mean():.4f}")
    logger.info(f"    Std: {results_df['pred_ite_prob'].std():.4f}")
    logger.info(f"    Min: {results_df['pred_ite_prob'].min():.4f}")
    logger.info(f"    Max: {results_df['pred_ite_prob'].max():.4f}")
    logger.info(f"  Mean predicted propensity: {results_df['pred_propensity_prob'].mean():.4f}")


def _save_filter_interpretations(
    model: CausalText,
    train_texts: List[str],
    output_dir: Path,
    top_k: int = 10
) -> None:
    """
    Generate and save filter interpretation analysis.

    Saves both a JSON file with structured data and a text summary.
    Only applicable for CNN feature extractors.

    Args:
        model: Trained CausalText model
        train_texts: Training texts to analyze activations on
        output_dir: Directory to save interpretation files
        top_k: Number of top n-grams per filter
    """
    # Filter interpretations only available for CNN models
    if model.feature_extractor_type != "cnn":
        logger.info("Filter interpretations not available for BERT feature extractor (CNN-only feature)")
        return

    logger.info(f"Analyzing filter activations on {len(train_texts)} texts...")

    # Get structured interpretations
    interpretations = model.feature_extractor.interpret_filters(
        train_texts,
        top_k=top_k,
        batch_size=32
    )

    # Save JSON with full data
    json_path = output_dir / "filter_interpretations.json"
    with open(json_path, 'w') as f:
        json.dump(interpretations, f, indent=2)
    logger.info(f"Filter interpretations saved to: {json_path}")

    # Save human-readable summary
    summary = model.feature_extractor.get_filter_summary(
        train_texts,
        top_k=top_k,
        batch_size=32
    )
    summary_path = output_dir / "filter_interpretations_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(summary)
    logger.info(f"Filter interpretation summary saved to: {summary_path}")


def _save_confounder_interpretations(
    model: CausalText,
    train_texts: List[str],
    output_dir: Path,
    top_k: int = 5,
    max_samples: int = 100
) -> None:
    """
    Generate and save confounder attention interpretation analysis.

    Saves JSON files with structured data showing:
    - Which sentences each confounder attends to (sentence-level weights)
    - How confounders are weighted for propensity vs outcome tasks (task-specific aggregation)

    Only applicable for confounder feature extractors.

    Args:
        model: Trained CausalText model
        train_texts: Training texts to analyze
        output_dir: Directory to save interpretation files
        top_k: Number of top-attended sentences per confounder
        max_samples: Maximum number of samples to analyze (for efficiency)
    """
    # Confounder interpretations only available for confounder extractors
    if model.feature_extractor_type != "confounder":
        logger.info("Confounder interpretations not available for this feature extractor type")
        return

    # Limit samples for efficiency
    texts_to_analyze = train_texts[:max_samples]
    logger.info(f"Analyzing confounder attention on {len(texts_to_analyze)} texts...")

    # Get interpretations from the feature extractor
    interpretations = model.feature_extractor.interpret_attention(
        texts_to_analyze,
        top_k=top_k
    )

    # Save interpretations JSON
    json_path = output_dir / "confounder_interpretations.json"
    with open(json_path, 'w') as f:
        json.dump(interpretations, f, indent=2, default=str)
    logger.info(f"Confounder interpretations saved to: {json_path}")

    # Get task-specific aggregation weights if available (for a sample batch)
    try:
        _, attention_info = model.feature_extractor(texts_to_analyze[:10], return_attention=True)

        # Extract task-specific confounder weights (3-way aggregation)
        task_weights = []
        for i, info in enumerate(attention_info):
            sample_info = {'sample_idx': i}
            if 'propensity_confounder_weights' in info:
                sample_info['propensity_weights'] = info['propensity_confounder_weights'].tolist()
            # DragonNet weights: y0, y1
            if 'y0_confounder_weights' in info:
                sample_info['y0_weights'] = info['y0_confounder_weights'].tolist()
            if 'y1_confounder_weights' in info:
                sample_info['y1_weights'] = info['y1_confounder_weights'].tolist()
            # R-Learner weights: outcome, tau
            if 'outcome_confounder_weights' in info:
                sample_info['outcome_weights'] = info['outcome_confounder_weights'].tolist()
            if 'tau_confounder_weights' in info:
                sample_info['tau_weights'] = info['tau_confounder_weights'].tolist()
            if len(sample_info) > 1:
                task_weights.append(sample_info)

        if task_weights:
            task_weights_path = output_dir / "confounder_task_weights.json"
            with open(task_weights_path, 'w') as f:
                json.dump(task_weights, f, indent=2)
            logger.info(f"Task-specific confounder weights saved to: {task_weights_path}")

    except Exception as e:
        logger.warning(f"Could not extract task-specific weights: {e}")

    # Generate human-readable summary
    summary_lines = [
        "=" * 80,
        "CONFOUNDER ATTENTION INTERPRETATION SUMMARY",
        "=" * 80,
        f"\nAnalyzed {len(texts_to_analyze)} documents",
        f"Top {top_k} attended sentences shown per confounder\n",
    ]

    # Aggregate across documents to find common patterns
    confounder_patterns = {}
    for doc_interp in interpretations:
        for conf_name, attended_sentences in doc_interp.items():
            if conf_name not in confounder_patterns:
                confounder_patterns[conf_name] = []
            for sent_info in attended_sentences:
                confounder_patterns[conf_name].append(sent_info)

    for conf_name in sorted(confounder_patterns.keys()):
        summary_lines.append(f"\n--- {conf_name} ---")
        attended = confounder_patterns[conf_name]
        # Sort by attention weight and show unique sentences
        seen = set()
        for info in sorted(attended, key=lambda x: -x['attention'])[:10]:
            sent = info['sentence'][:100]
            if sent not in seen:
                summary_lines.append(f"  [{info['attention']:.3f}] {sent}...")
                seen.add(sent)

    summary_path = output_dir / "confounder_interpretations_summary.txt"
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    logger.info(f"Confounder interpretation summary saved to: {summary_path}")
