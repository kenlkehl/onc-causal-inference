#!/usr/bin/env python
"""Oracle runner for the R-learner representation -> causal forest X/W split.

This is a narrowed variant of run_oracle_experiments.py.  It only evaluates the
two-stage CausalTextForest path where Stage 1 trains an R-learner representation
and Stage 2 fits EconML CausalForestDML with separate X and W matrices:

- X: effect-modifier branch activations plus explicit features with the
  "effect_modifier" role.
- W: nuisance branch activations plus explicit features with the "confounder"
  role.

LLM hidden-state downprojection is intentionally disabled for the LLM-based
extractors: flp_downprojection_dim=None and hlm_downprojection_dim=None.

Output is compatible with the existing oracle result directory layout.
"""

import argparse
import gc
import hashlib
import itertools
import json
import logging
import multiprocessing as mp
import os
import queue
import random
import threading
import traceback
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from oci.config import ExplicitFeatureSpec, TRAINABLE_EXTRACTOR_TYPES
from oci.data import (
    CachedHiddenStateDataset,
    ClinicalTextDataset,
    collate_batch,
    collate_cached_batch,
    prepare_cached_batch,
)
from oci.models.causal_text_forest import CausalTextForest
from oci.models.gpu_hidden_state_store import GPUHiddenStateStore
from oci.models.hidden_state_cache import HiddenStateCache

from run_oracle_experiments import (
    _common_model_kwargs,
    _get_cache_info,
    _open_cache_for_worker,
    _resolve_parquet_file,
    compute_metrics,
    group_configs_by_cache_key,
    load_single_gpu_store,
    precompute_single_cache,
    resolve_workers_per_gpu,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class XWRLearnerForestConfig:
    """Configuration for one X/W R-learner causal forest experiment."""

    dataset_path: str
    dataset_name: str

    # Fixed by this runner. Kept in the serialized config for analyzer parity.
    model_type: str = "causal_forest"
    rlearner_mode: str = "xw_branch"
    xw_feature_split: bool = True
    use_explicit_features: bool = False
    # Compatibility alias for older analysis scripts.
    use_explicit_confounders: bool = False

    feature_extractor_type: str = "frozen_llm_pooler"
    repeat_index: int = 0

    # Frozen LLM Pooler hyperparameters.
    flp_max_length: int = 10000
    flp_freeze_llm: bool = True
    flp_projection_dim: int = 128
    flp_gated_attention_dim: int = 128
    flp_downprojection_dim: Optional[int] = None
    flp_cache_hidden_states: bool = False
    flp_chat_template_prompt: Optional[str] = None
    flp_model_name: str = "Qwen/Qwen3.5-0.8B-Base"
    flp_dropout: float = 0.1
    flp_gradient_checkpointing: bool = True

    # Fixed training parameters.
    epochs: int = 30
    batch_size: int = 2
    learning_rate: float = 1e-4
    n_folds: int = 5
    gamma_rlearner: float = 1.0

    # Causal forest parameters.
    cf_n_estimators: int = 200
    cf_min_samples_leaf: int = 5

    # Hierarchical LLM hyperparameters.
    hlm_model_name: str = "Qwen/Qwen3.5-0.8B-Base"
    hlm_chunk_size: int = 2048
    hlm_chunk_overlap: int = 256
    hlm_max_chunks: int = 16
    hlm_downprojection_dim: Optional[int] = None
    hlm_freeze_llm: bool = True
    hlm_cache_hidden_states: bool = False
    hlm_chat_template_prompt: Optional[str] = None

    # Hierarchical CNN hyperparameters.
    hcnn_embedding_dim: int = 256
    hcnn_conv_dim: int = 256
    hcnn_kernel_size: int = 5
    hcnn_num_conv_blocks: int = 4
    hcnn_chunk_size: int = 12000
    hcnn_chunk_overlap: int = 64
    hcnn_max_chunks: int = 32
    hcnn_vocab_size: int = 50000
    hcnn_projection_dim: int = 128
    hcnn_dropout: float = 0.1

    # Hierarchical GRU hyperparameters.
    hgru_embedding_dim: int = 256
    hgru_gru_hidden_dim: int = 256
    hgru_num_gru_layers: int = 2
    hgru_chunk_size: int = 12000
    hgru_chunk_overlap: int = 64
    hgru_max_chunks: int = 32
    hgru_vocab_size: int = 50000
    hgru_projection_dim: int = 128
    hgru_dropout: float = 0.1

    # Simple CNN hyperparameters.
    scnn_embedding_dim: int = 256
    scnn_conv_dim: int = 256
    scnn_kernel_size: int = 5
    scnn_num_conv_blocks: int = 4
    scnn_max_length: int = 20000
    scnn_vocab_size: int = 50000
    scnn_projection_dim: int = 128
    scnn_dropout: float = 0.1

    _EXTRACTOR_PREFIXES = {
        "frozen_llm_pooler": {"flp_"},
        "hierarchical_llm": {"hlm_"},
        "hierarchical_cnn": {"hcnn_"},
        "hierarchical_gru": {"hgru_"},
        "simple_cnn": {"scnn_"},
    }
    _ALL_EXTRACTOR_PREFIXES = set().union(*_EXTRACTOR_PREFIXES.values())

    def __post_init__(self):
        self.model_type = "causal_forest"
        self.rlearner_mode = "xw_branch"
        self.xw_feature_split = True
        self.use_explicit_confounders = self.use_explicit_features
        self.flp_downprojection_dim = None
        self.hlm_downprojection_dim = None

    def config_hash(self) -> str:
        d = asdict(self)
        keep_prefixes = self._EXTRACTOR_PREFIXES.get(
            self.feature_extractor_type, set()
        )
        remove_prefixes = self._ALL_EXTRACTOR_PREFIXES - keep_prefixes
        d = {
            k: v for k, v in d.items()
            if not any(k.startswith(p) for p in remove_prefixes)
        }
        config_str = json.dumps(d, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]


def _feature_key(spec: ExplicitFeatureSpec) -> Tuple[str, str]:
    categories = ",".join(spec.categories or [])
    return (spec.name, categories)


def _metadata_entry_to_spec(
    entry: Dict[str, Any],
    default_roles: Optional[List[str]] = None,
) -> ExplicitFeatureSpec:
    roles = entry.get("roles") or default_roles or ["confounder"]
    return ExplicitFeatureSpec(
        name=entry["name"],
        type=entry["type"],
        categories=entry.get("categories"),
        description=entry.get("description"),
        roles=list(roles),
    )


def load_explicit_feature_specs_from_metadata(
    dataset_path: str,
) -> List[ExplicitFeatureSpec]:
    """Load role-tagged explicit feature specs from metadata.json.

    New datasets should provide metadata["features"] with roles.  For older
    datasets, metadata["confounders"] is treated as confounder-role features,
    and metadata["effect_modifiers"] is also honored if present.
    """
    metadata_file = Path(dataset_path) / "metadata.json"
    if not metadata_file.exists():
        logger.warning("metadata.json not found at %s", metadata_file)
        return []

    with open(metadata_file) as f:
        metadata = json.load(f)

    roles_by_name: Dict[str, List[str]] = {}
    for key, role in (("confounders", "confounder"), ("effect_modifiers", "effect_modifier")):
        for entry in metadata.get(key, []):
            name = entry["name"]
            roles_by_name.setdefault(name, [])
            if role not in roles_by_name[name]:
                roles_by_name[name].append(role)

    specs: List[ExplicitFeatureSpec] = []
    if metadata.get("features"):
        for entry in metadata["features"]:
            specs.append(
                _metadata_entry_to_spec(
                    entry,
                    default_roles=roles_by_name.get(entry["name"]),
                )
            )
    else:
        merged: Dict[str, Dict[str, Any]] = {}
        for key, role in (("confounders", "confounder"), ("effect_modifiers", "effect_modifier")):
            for entry in metadata.get(key, []):
                name = entry["name"]
                if name not in merged:
                    merged[name] = dict(entry)
                    merged[name]["roles"] = []
                if role not in merged[name]["roles"]:
                    merged[name]["roles"].append(role)
        specs = [_metadata_entry_to_spec(entry) for entry in merged.values()]

    seen = set()
    unique_specs = []
    for spec in specs:
        key = _feature_key(spec)
        if key in seen:
            continue
        seen.add(key)
        unique_specs.append(spec)
    return unique_specs


def prepare_explicit_feature_columns(
    df: pd.DataFrame,
    specs: List[ExplicitFeatureSpec],
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Normalize explicit feature columns to explicit_feat_* names."""
    if not specs:
        return df, [], []

    df = df.copy()
    feature_cols = []
    missing = []

    for spec in specs:
        target = f"explicit_feat_{spec.name}"
        candidates = [
            target,
            f"explicit_conf_{spec.name}",
            f"llm_extracted_{spec.name}",
        ]
        source = next((col for col in candidates if col in df.columns), None)
        if source is None:
            missing.append(target)
            continue

        if source != target:
            df[target] = df[source]

        source_missing = f"{source}_missing"
        target_missing = f"{target}_missing"
        if source_missing in df.columns and target_missing not in df.columns:
            df[target_missing] = df[source_missing]

        feature_cols.append(target)

    return df, feature_cols, missing


def _create_datasets_and_loaders(
    train_df,
    test_df,
    train_idx,
    test_idx,
    text_column,
    explicit_feature_cols,
    batch_size,
    hidden_state_cache,
    gpu_store,
):
    """Create train/test datasets and DataLoaders with optional hidden-state cache."""
    use_cache = hidden_state_cache is not None
    if use_cache:
        chunk_counts = hidden_state_cache.chunk_counts
    elif gpu_store is not None:
        chunk_counts = gpu_store.chunk_counts
    else:
        chunk_counts = None

    if gpu_store is not None:
        train_dataset = CachedHiddenStateDataset(
            data=train_df,
            text_column=text_column,
            outcome_column="outcome_indicator",
            treatment_column="treatment_indicator",
            dataset_indices=np.array(train_idx),
            explicit_feature_columns=explicit_feature_cols,
            cache_chunk_counts=chunk_counts,
        )
        test_dataset = CachedHiddenStateDataset(
            data=test_df,
            text_column=text_column,
            outcome_column="outcome_indicator",
            treatment_column="treatment_indicator",
            dataset_indices=np.array(test_idx),
            explicit_feature_columns=explicit_feature_cols,
            cache_chunk_counts=chunk_counts,
        )
        collate_fn = collate_cached_batch
    elif use_cache:
        train_dataset = CachedHiddenStateDataset(
            data=train_df,
            text_column=text_column,
            outcome_column="outcome_indicator",
            treatment_column="treatment_indicator",
            dataset_indices=np.array(train_idx),
            explicit_feature_columns=explicit_feature_cols,
            cache_hidden_states=hidden_state_cache.hidden_states_array,
            cache_attention_masks=hidden_state_cache.attention_mask_array,
            cache_chunk_counts=chunk_counts,
        )
        test_dataset = CachedHiddenStateDataset(
            data=test_df,
            text_column=text_column,
            outcome_column="outcome_indicator",
            treatment_column="treatment_indicator",
            dataset_indices=np.array(test_idx),
            explicit_feature_columns=explicit_feature_cols,
            cache_hidden_states=hidden_state_cache.hidden_states_array,
            cache_attention_masks=hidden_state_cache.attention_mask_array,
            cache_chunk_counts=chunk_counts,
        )
        collate_fn = collate_cached_batch
    else:
        train_dataset = ClinicalTextDataset(
            data=train_df,
            text_column=text_column,
            outcome_column="outcome_indicator",
            treatment_column="treatment_indicator",
            explicit_feature_columns=explicit_feature_cols,
        )
        test_dataset = ClinicalTextDataset(
            data=test_df,
            text_column=text_column,
            outcome_column="outcome_indicator",
            treatment_column="treatment_indicator",
            explicit_feature_columns=explicit_feature_cols,
        )
        collate_fn = collate_batch

    if gpu_store is not None:
        dl_kwargs = dict(num_workers=0)
    elif use_cache:
        dl_kwargs = dict(num_workers=2, persistent_workers=True, pin_memory=True)
    else:
        dl_kwargs = dict(
            num_workers=2,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=2,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        **dl_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        **dl_kwargs,
    )
    return train_dataset, test_dataset, train_loader, test_loader, collate_fn, dl_kwargs


def _make_combined_loader(
    combined_df,
    combined_indices,
    text_column,
    explicit_feature_cols,
    batch_size,
    hidden_state_cache,
    gpu_store,
    dl_kwargs,
):
    """Build the combined loader used by the causal forest stage."""
    if hidden_state_cache is not None:
        chunk_counts = hidden_state_cache.chunk_counts
    elif gpu_store is not None:
        chunk_counts = gpu_store.chunk_counts
    else:
        chunk_counts = None

    if gpu_store is not None:
        dataset = CachedHiddenStateDataset(
            data=combined_df,
            text_column=text_column,
            outcome_column="outcome_indicator",
            treatment_column="treatment_indicator",
            dataset_indices=combined_indices,
            explicit_feature_columns=explicit_feature_cols,
            cache_chunk_counts=chunk_counts,
        )
        collate_fn = collate_cached_batch
    elif hidden_state_cache is not None:
        dataset = CachedHiddenStateDataset(
            data=combined_df,
            text_column=text_column,
            outcome_column="outcome_indicator",
            treatment_column="treatment_indicator",
            dataset_indices=combined_indices,
            explicit_feature_columns=explicit_feature_cols,
            cache_hidden_states=hidden_state_cache.hidden_states_array,
            cache_attention_masks=hidden_state_cache.attention_mask_array,
            cache_chunk_counts=chunk_counts,
        )
        collate_fn = collate_cached_batch
    else:
        dataset = ClinicalTextDataset(
            data=combined_df,
            text_column=text_column,
            outcome_column="outcome_indicator",
            treatment_column="treatment_indicator",
            explicit_feature_columns=explicit_feature_cols,
        )
        collate_fn = collate_batch

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        **dl_kwargs,
    )


def run_xw_rlearner_forest_experiment(
    config: XWRLearnerForestConfig,
    device: torch.device,
    df: pd.DataFrame,
    explicit_feature_specs: List[ExplicitFeatureSpec],
    explicit_feature_cols: Optional[List[str]],
    gpu_store,
    hidden_state_cache,
) -> Dict[str, Any]:
    """Run K-fold CV for the R-learner representation -> causal forest path."""
    text_column = "clinical_text"
    batch_size = config.batch_size
    df = df.reset_index(drop=True)
    kf = KFold(n_splits=config.n_folds, shuffle=True, random_state=42 + config.repeat_index)

    all_predictions = []
    use_cached = gpu_store is not None or hidden_state_cache is not None

    for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        model_kwargs = _common_model_kwargs(
            config,
            gpu_store,
            hidden_state_cache,
            explicit_feature_specs,
            device,
        )
        model_kwargs.update(
            dict(
                representation_dim=128,
                hidden_dim=64,
                dropout=0.2,
                cf_n_estimators=config.cf_n_estimators,
                cf_min_samples_leaf=config.cf_min_samples_leaf,
                cf_honest=True,
                cf_inference=True,
                cf_use_rlearner_representation=True,
                cf_gamma_rlearner=config.gamma_rlearner,
                cf_rlearner_dual_extractors=False,
                explicit_feature_specs=explicit_feature_specs,
            )
        )

        model = CausalTextForest(**model_kwargs)

        if config.feature_extractor_type in TRAINABLE_EXTRACTOR_TYPES:
            model.fit_tokenizer(train_df[text_column].tolist())

        for name, param in model.named_parameters():
            if param.dtype != torch.float32:
                logger.warning(
                    "Parameter %s has dtype %s; casting to float32",
                    name,
                    param.dtype,
                )
                param.data = param.data.float()

        (
            train_dataset,
            _test_dataset,
            train_loader,
            test_loader,
            _collate_fn,
            dl_kwargs,
        ) = _create_datasets_and_loaders(
            train_df,
            test_df,
            train_idx,
            test_idx,
            text_column,
            explicit_feature_cols,
            batch_size,
            hidden_state_cache,
            gpu_store,
        )

        if explicit_feature_specs and train_dataset.explicit_feature_values:
            model.fit_explicit_features(train_dataset.explicit_feature_values)
            model.fit_explicit_feature_featurizer(train_dataset.explicit_feature_values)

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.learning_rate,
            weight_decay=0.01,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.epochs,
        )

        best_val_loss = float("inf")
        best_state = None

        for _epoch in range(config.epochs):
            model.train()
            for batch in train_loader:
                batch["treatment"] = batch["treatment"].to(device)
                batch["outcome"] = batch["outcome"].to(device)
                if use_cached:
                    prepare_cached_batch(batch, device, gpu_store=gpu_store)

                optimizer.zero_grad()
                losses = model.train_representation_step(
                    batch,
                    alpha_propensity=1.0,
                    gamma_rlearner=config.gamma_rlearner,
                )
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    1.0,
                )
                optimizer.step()

            scheduler.step()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in test_loader:
                    batch["treatment"] = batch["treatment"].to(device)
                    batch["outcome"] = batch["outcome"].to(device)
                    if use_cached:
                        prepare_cached_batch(batch, device, gpu_store=gpu_store)
                    losses = model.train_representation_step(
                        batch,
                        alpha_propensity=1.0,
                        gamma_rlearner=config.gamma_rlearner,
                    )
                    val_loss += losses["loss"].item()

            val_loss /= len(test_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if best_state:
            model.load_state_dict(best_state)
            model.to(device)

        combined_df = pd.concat([train_df, test_df])
        combined_indices = np.concatenate([train_idx, test_idx])
        combined_loader = _make_combined_loader(
            combined_df,
            combined_indices,
            text_column,
            explicit_feature_cols,
            batch_size,
            hidden_state_cache,
            gpu_store,
            dl_kwargs,
        )
        combined_T = combined_df["treatment_indicator"].values
        combined_Y = combined_df["outcome_indicator"].values

        cf_kwargs = dict(gpu_store=gpu_store) if use_cached else {}
        model.train_causal_forest(combined_loader, combined_T, combined_Y, **cf_kwargs)
        preds = model.predict(test_loader, return_ci=True, **cf_kwargs)

        fold_preds = test_df.copy()
        fold_preds["pred_y0_prob"] = preds["pred_y0_prob"]
        fold_preds["pred_y1_prob"] = preds["pred_y1_prob"]
        fold_preds["pred_ite_prob"] = preds["pred_ite_prob"]
        fold_preds["pred_propensity"] = preds["propensity_prob"]
        fold_preds["pred_tau"] = preds["tau_pred"]
        fold_preds["cv_fold"] = fold + 1
        if "tau_lower" in preds:
            fold_preds["pred_tau_lower"] = preds["tau_lower"]
            fold_preds["pred_tau_upper"] = preds["tau_upper"]

        all_predictions.append(fold_preds)

        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results_df = pd.concat(all_predictions).sort_index()
    metrics = compute_metrics(
        pred_ite=results_df["pred_ite_prob"].values,
        true_ite=results_df["true_ite_prob"].values,
        pred_propensity=results_df["pred_propensity"].values,
        true_treatment=results_df["treatment_indicator"].values,
        pred_y0=results_df["pred_y0_prob"].values,
        pred_y1=results_df["pred_y1_prob"].values,
        true_y0=results_df["true_y0_prob"].values,
        true_y1=results_df["true_y1_prob"].values,
        true_outcome=results_df["outcome_indicator"].values,
        tau_lower=(
            results_df["pred_tau_lower"].values
            if "pred_tau_lower" in results_df.columns
            else None
        ),
        tau_upper=(
            results_df["pred_tau_upper"].values
            if "pred_tau_upper" in results_df.columns
            else None
        ),
    )
    return {"metrics": metrics, "n_samples": len(results_df)}


def run_single_experiment(
    config: XWRLearnerForestConfig,
    device: str,
    output_dir: Path,
    cache_registry: Optional[Dict[str, HiddenStateCache]] = None,
    gpu_store_registry: Optional[Dict[str, GPUHiddenStateStore]] = None,
) -> Dict[str, Any]:
    """Run a single experiment configuration."""
    del output_dir
    device_obj = torch.device(device)

    parquet_file = _resolve_parquet_file(config.dataset_path)
    if parquet_file is None:
        return {"error": f"Dataset not found in {config.dataset_path}", "skipped": True}

    df = pd.read_parquet(parquet_file)
    if "clinical_text" not in df.columns:
        return {"error": "Text column 'clinical_text' not found", "skipped": True}

    explicit_feature_specs: List[ExplicitFeatureSpec] = []
    explicit_feature_cols = None
    if config.use_explicit_features:
        explicit_feature_specs = load_explicit_feature_specs_from_metadata(config.dataset_path)
        if not explicit_feature_specs:
            return {
                "error": f"No explicit feature specs found in {config.dataset_path}",
                "skipped": True,
            }

        df, explicit_feature_cols, missing_cols = prepare_explicit_feature_columns(
            df,
            explicit_feature_specs,
        )
        if missing_cols:
            return {
                "error": (
                    "Explicit feature columns missing from dataset: "
                    f"{missing_cols}. Expected explicit_feat_*, explicit_conf_*, "
                    "or llm_extracted_* columns."
                ),
                "skipped": True,
            }

        role_counts = {"confounder": 0, "effect_modifier": 0, "both": 0}
        for spec in explicit_feature_specs:
            role_set = set(spec.roles)
            if role_set == {"confounder", "effect_modifier"}:
                role_counts["both"] += 1
            else:
                for role in role_set:
                    role_counts[role] += 1
        logger.info(
            "Using %d role-tagged explicit features: %s",
            len(explicit_feature_specs),
            role_counts,
        )

    gpu_store, hidden_state_cache = _get_cache_info(
        config,
        parquet_file,
        cache_registry,
        gpu_store_registry,
    )

    result = run_xw_rlearner_forest_experiment(
        config,
        device_obj,
        df,
        explicit_feature_specs,
        explicit_feature_cols,
        gpu_store,
        hidden_state_cache,
    )

    return {
        "config": asdict(config),
        "metrics": result["metrics"],
        "n_samples": result["n_samples"],
        "skipped": False,
        "error": None,
    }


def generate_experiment_grid(
    dataset_paths: List[str],
    filter_max_lengths: Optional[List[int]] = None,
    model_names: Optional[List[str]] = None,
    chat_template_prompt: Optional[str] = None,
    filter_extractor_types: Optional[List[str]] = None,
    learning_rates: Optional[List[float]] = None,
    epoch_counts: Optional[List[int]] = None,
    include_explicit_feature_options: Optional[List[bool]] = None,
) -> List[XWRLearnerForestConfig]:
    """Generate the narrowed experiment grid."""
    if model_names is None:
        model_names = [
            "Qwen/Qwen3.5-0.8B-Base",
            "Qwen/Qwen3.5-0.8B",
            "google/medgemma-1.5-4b-it",
        ]
    if learning_rates is None:
        learning_rates = [1e-5, 1e-4]
    if epoch_counts is None:
        epoch_counts = [5, 10, 25, 50]
    if include_explicit_feature_options is None:
        include_explicit_feature_options = [False, True]

    datasets = [(p, Path(p).name) for p in dataset_paths]
    all_extractor_types = [
        "frozen_llm_pooler",
        "hierarchical_llm",
        "hierarchical_cnn",
        "hierarchical_gru",
        "simple_cnn",
    ]
    extractor_types = all_extractor_types
    if filter_extractor_types:
        extractor_types = [e for e in all_extractor_types if e in filter_extractor_types]

    configs: List[XWRLearnerForestConfig] = []

    for ext_type in extractor_types:
        if ext_type == "frozen_llm_pooler":
            max_lengths = [100000]
            if filter_max_lengths:
                max_lengths = [m for m in max_lengths if m in filter_max_lengths]
            chat_template_options = [None]
            if chat_template_prompt is not None:
                chat_template_options = [None, chat_template_prompt]

            for (
                dataset_path,
                dataset_name,
            ), max_len, use_feats, ctp, mn, lr, ep in itertools.product(
                datasets,
                max_lengths,
                include_explicit_feature_options,
                chat_template_options,
                model_names,
                learning_rates,
                epoch_counts,
            ):
                configs.append(
                    XWRLearnerForestConfig(
                        dataset_path=dataset_path,
                        dataset_name=dataset_name,
                        use_explicit_features=use_feats,
                        feature_extractor_type="frozen_llm_pooler",
                        flp_max_length=max_len,
                        flp_downprojection_dim=None,
                        flp_model_name=mn,
                        flp_chat_template_prompt=ctp,
                        learning_rate=lr,
                        epochs=ep,
                    )
                )

        elif ext_type == "hierarchical_llm":
            chunk_size = 2048
            chunk_overlap = 256
            max_chunks_options = [4, 8, 16]

            for (
                dataset_path,
                dataset_name,
            ), n_chunks, use_feats, mn, lr, ep in itertools.product(
                datasets,
                max_chunks_options,
                include_explicit_feature_options,
                model_names,
                learning_rates,
                epoch_counts,
            ):
                configs.append(
                    XWRLearnerForestConfig(
                        dataset_path=dataset_path,
                        dataset_name=dataset_name,
                        use_explicit_features=use_feats,
                        feature_extractor_type="hierarchical_llm",
                        hlm_model_name=mn,
                        hlm_chunk_size=chunk_size,
                        hlm_chunk_overlap=chunk_overlap,
                        hlm_max_chunks=n_chunks,
                        hlm_downprojection_dim=None,
                        learning_rate=lr,
                        epochs=ep,
                    )
                )

        elif ext_type == "hierarchical_cnn":
            chunk_sizes = [256, 512]
            for (
                dataset_path,
                dataset_name,
            ), use_feats, cs, lr, ep in itertools.product(
                datasets,
                include_explicit_feature_options,
                chunk_sizes,
                learning_rates,
                epoch_counts,
            ):
                configs.append(
                    XWRLearnerForestConfig(
                        dataset_path=dataset_path,
                        dataset_name=dataset_name,
                        use_explicit_features=use_feats,
                        feature_extractor_type="hierarchical_cnn",
                        hcnn_chunk_size=cs,
                        learning_rate=lr,
                        epochs=ep,
                    )
                )

        elif ext_type == "hierarchical_gru":
            chunk_sizes = [256, 512]
            for (
                dataset_path,
                dataset_name,
            ), use_feats, cs, lr, ep in itertools.product(
                datasets,
                include_explicit_feature_options,
                chunk_sizes,
                learning_rates,
                epoch_counts,
            ):
                configs.append(
                    XWRLearnerForestConfig(
                        dataset_path=dataset_path,
                        dataset_name=dataset_name,
                        use_explicit_features=use_feats,
                        feature_extractor_type="hierarchical_gru",
                        hgru_chunk_size=cs,
                        learning_rate=lr,
                        epochs=ep,
                    )
                )

        elif ext_type == "simple_cnn":
            scnn_max_lengths = [5000, 10000, 25000]
            if filter_max_lengths:
                scnn_max_lengths = [m for m in scnn_max_lengths if m in filter_max_lengths]

            for (
                dataset_path,
                dataset_name,
            ), use_feats, max_len, lr, ep in itertools.product(
                datasets,
                include_explicit_feature_options,
                scnn_max_lengths,
                learning_rates,
                epoch_counts,
            ):
                configs.append(
                    XWRLearnerForestConfig(
                        dataset_path=dataset_path,
                        dataset_name=dataset_name,
                        use_explicit_features=use_feats,
                        feature_extractor_type="simple_cnn",
                        scnn_max_length=max_len,
                        learning_rate=lr,
                        epochs=ep,
                    )
                )

    random.Random(42).shuffle(configs)
    return configs


def _is_real_cache_group(cache_hash: str, cache_info: Optional[dict]) -> bool:
    return bool(cache_info) and not cache_hash.startswith("__no_cache__")


def worker_process_fn(
    device: str,
    job_queue: mp.Queue,
    progress_queue: mp.Queue,
    output_dir: str,
    cache_hash: str,
    cache_info: Optional[dict],
    use_gpu_cache: bool,
    cache_base_dir: Optional[str] = None,
):
    """Worker process for cached LLM experiments."""
    output_dir_path = Path(output_dir)
    torch.set_default_dtype(torch.float32)

    cache_registry = {}
    gpu_store_registry = {}

    if _is_real_cache_group(cache_hash, cache_info):
        cache = _open_cache_for_worker(cache_hash, cache_info, cache_base_dir=cache_base_dir)
        cache_registry[cache_hash] = cache
        if use_gpu_cache:
            store = load_single_gpu_store(cache, cache_info, device)
            if store is not None:
                gpu_store_registry = {cache_hash: store}

    logger.info("Worker process started on %s (pid=%s)", device, os.getpid())

    while True:
        try:
            config = job_queue.get(timeout=2)
        except Exception:
            break

        config_hash = config.config_hash()
        try:
            result = run_single_experiment(
                config,
                device,
                output_dir_path,
                cache_registry,
                gpu_store_registry,
            )
            result_file = output_dir_path / "results" / f"{config_hash}.json"
            result_file.parent.mkdir(parents=True, exist_ok=True)
            with open(result_file, "w") as f:
                json.dump(result, f, indent=2, default=str)
            progress_queue.put(("done", config_hash, result))
        except Exception as e:
            tb = traceback.format_exc()
            logger.error("Experiment %s FAILED: %s\n%s", config_hash, e, tb)
            error_result = {
                "config": asdict(config),
                "error": str(e),
                "skipped": True,
            }
            result_file = output_dir_path / "results" / f"{config_hash}.json"
            result_file.parent.mkdir(parents=True, exist_ok=True)
            with open(result_file, "w") as f:
                json.dump(error_result, f, indent=2, default=str)
            progress_queue.put(("error", config_hash, error_result))

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for store in gpu_store_registry.values():
        store.free()
    for cache in cache_registry.values():
        cache.close()
    logger.info("Worker process on %s (pid=%s) finished", device, os.getpid())


def worker_thread(
    device: str,
    job_queue: queue.Queue,
    results_dict: Dict[str, Any],
    output_dir: Path,
    lock: threading.Lock,
    progress_bar: tqdm,
):
    """Thread worker for live LLM or trainable non-cache groups."""
    while True:
        try:
            config = job_queue.get(timeout=1)
        except queue.Empty:
            break

        config_hash = config.config_hash()
        try:
            result = run_single_experiment(config, device, output_dir, {}, {})
            with lock:
                results_dict[config_hash] = result
                result_file = output_dir / "results" / f"{config_hash}.json"
                result_file.parent.mkdir(parents=True, exist_ok=True)
                with open(result_file, "w") as f:
                    json.dump(result, f, indent=2, default=str)
                progress_bar.update(1)
                if result.get("skipped"):
                    progress_bar.set_postfix_str(
                        f"Skipped: {result.get('error', 'unknown')[:30]}"
                    )
                else:
                    metrics = result.get("metrics", {})
                    progress_bar.set_postfix_str(
                        f"X/W RF-CF ITE corr: {metrics.get('ite_corr', float('nan')):.3f}"
                    )
        except Exception as e:
            tb = traceback.format_exc()
            logger.error("Experiment %s FAILED: %s\n%s", config_hash, e, tb)
            with lock:
                error_result = {
                    "config": asdict(config),
                    "error": str(e),
                    "skipped": True,
                }
                results_dict[config_hash] = error_result
                result_file = output_dir / "results" / f"{config_hash}.json"
                result_file.parent.mkdir(parents=True, exist_ok=True)
                with open(result_file, "w") as f:
                    json.dump(error_result, f, indent=2, default=str)
                progress_bar.update(1)
                progress_bar.set_postfix_str(f"Error: {str(e)[:50]}")
        finally:
            job_queue.task_done()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _parse_bool_grid(values: Optional[List[str]]) -> Optional[List[bool]]:
    if values is None:
        return None
    parsed = []
    for value in values:
        lowered = value.lower()
        if lowered in {"true", "1", "yes", "y"}:
            parsed.append(True)
        elif lowered in {"false", "0", "no", "n"}:
            parsed.append(False)
        else:
            raise argparse.ArgumentTypeError(
                f"Boolean grid values must be true/false, got {value}"
            )
    return parsed


def print_grid_summary(
    pending_configs: List[XWRLearnerForestConfig],
    completed_count: int,
    n_repeats: int,
    base_config_count: int,
):
    model_type_summary = {}
    extractor_summary = {}
    llm_model_summary = {}
    for config in pending_configs:
        model_type_summary[config.model_type] = model_type_summary.get(config.model_type, 0) + 1
        extractor_summary[config.feature_extractor_type] = (
            extractor_summary.get(config.feature_extractor_type, 0) + 1
        )
        if config.feature_extractor_type == "frozen_llm_pooler":
            llm_model_summary[config.flp_model_name] = (
                llm_model_summary.get(config.flp_model_name, 0) + 1
            )
        elif config.feature_extractor_type == "hierarchical_llm":
            llm_model_summary[config.hlm_model_name] = (
                llm_model_summary.get(config.hlm_model_name, 0) + 1
            )

    dataset_names = sorted(set(config.dataset_name for config in pending_configs))
    lr_values = sorted(set(config.learning_rate for config in pending_configs))
    epoch_values = sorted(set(config.epochs for config in pending_configs))
    explicit_values = sorted(set(config.use_explicit_features for config in pending_configs))

    print(f"\n{'=' * 60}")
    print("X/W R-Learner -> Causal Forest Grid Summary")
    print(f"{'=' * 60}")
    print(f"Base configs before repeats: {base_config_count}")
    print(f"Repeats: {n_repeats}")
    print(f"Total experiments to run: {len(pending_configs)}")
    if completed_count:
        print(f"Already completed (skipped): {completed_count}")
    print("Model path: causal_forest with cf_use_rlearner_representation=True")
    print("X/W split: enabled")
    print("LLM hidden-state downprojection: disabled")
    print(f"Model types: {', '.join(f'{k}({v})' for k, v in sorted(model_type_summary.items()))}")
    print(f"Extractors:  {', '.join(f'{k}({v})' for k, v in sorted(extractor_summary.items()))}")
    if llm_model_summary:
        print(f"LLMs:       {', '.join(f'{k}({v})' for k, v in sorted(llm_model_summary.items()))}")
    print(f"Datasets:   {', '.join(dataset_names)}")
    print(f"Explicit features: {', '.join(str(v) for v in explicit_values)}")
    print(f"LR values:  {', '.join(str(v) for v in lr_values)}")
    print(f"Epochs:     {', '.join(str(v) for v in epoch_values)}")
    print(f"{'=' * 60}")


def aggregate_results(output_dir: Path, results_dict: Dict[str, Any]):
    all_results = []
    for result in results_dict.values():
        if not result.get("skipped"):
            row = {**result.get("config", {}), **result.get("metrics", {})}
            all_results.append(row)

    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(output_dir / "all_results.csv", index=False)
        results_df.to_parquet(output_dir / "all_results.parquet", index=False)

        group_cols = [
            "dataset_name",
            "feature_extractor_type",
            "model_type",
            "rlearner_mode",
            "flp_model_name",
            "hlm_model_name",
            "flp_max_length",
            "use_explicit_features",
            "learning_rate",
            "epochs",
        ]
        group_cols = [col for col in group_cols if col in results_df.columns]
        metric_agg = {}
        for metric in [
            "ite_corr",
            "ite_spearman_corr",
            "ate_bias",
            "propensity_auroc",
            "ite_mse",
            "ite_mae",
            "ci_coverage",
            "mean_ci_width",
        ]:
            if metric in results_df.columns:
                metric_agg[metric] = ["mean", "std"]

        summary = results_df.groupby(group_cols).agg(metric_agg)
        summary.to_csv(output_dir / "summary.csv")
        logger.info("\nSummary (mean +/- std across repeats):\n%s", summary)

    logger.info(
        "Total experiments: %d, Successful: %d, Skipped/Failed: %d",
        len(results_dict),
        len(all_results),
        len(results_dict) - len(all_results),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Oracle runner for R-learner X/W activations into CausalForestDML"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="../pcori_experiments/oracle_xw_rlearner_forest",
        help="Output directory for results",
    )
    parser.add_argument(
        "--devices",
        type=str,
        nargs="+",
        default=["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
        help="GPU devices to use",
    )
    parser.add_argument(
        "--max-experiments",
        type=int,
        default=None,
        help="Maximum number of pending experiments to run",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from existing results")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved grid and exit without running experiments",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        required=True,
        help="Dataset directories containing dataset.parquet or dataset_with_extraction.parquet",
    )
    parser.add_argument(
        "--max-lengths",
        type=int,
        nargs="+",
        default=None,
        help="Filter max lengths for frozen_llm_pooler/simple_cnn grids",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of CV folds",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Opt in to pre-caching hidden states to disk",
    )
    parser.add_argument(
        "--gpu-cache",
        action="store_true",
        help="Keep pre-computed hidden states in GPU VRAM instead of disk cache",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=10,
        help="Number of repeats per base config",
    )
    parser.add_argument(
        "--model-names",
        type=str,
        nargs="+",
        default=[
            "Qwen/Qwen3.5-0.8B-Base",
            "Qwen/Qwen3.5-0.8B",
            "google/medgemma-1.5-4b-it",
        ],
        help="HuggingFace model names for LLM-based extractors",
    )
    parser.add_argument(
        "--chat-template-prompt",
        type=str,
        default=None,
        help="Optional chat template prompt for frozen_llm_pooler runs",
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=str,
        default="auto",
        help="Concurrent workers per GPU for cached LLM experiments: 'auto' or integer",
    )
    parser.add_argument(
        "--filter-extractor-types",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Feature extractors to include: frozen_llm_pooler, hierarchical_llm, "
            "hierarchical_cnn, hierarchical_gru, simple_cnn"
        ),
    )
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=[1e-5, 1e-4],
        help="Learning-rate grid",
    )
    parser.add_argument(
        "--epoch-counts",
        type=int,
        nargs="+",
        default=[5, 10, 25, 50],
        help="Epoch-count grid",
    )
    parser.add_argument(
        "--explicit-feature-options",
        type=str,
        nargs="+",
        default=None,
        help="Boolean grid for using role-tagged explicit features; default false true",
    )

    args = parser.parse_args()

    if args.workers_per_gpu != "auto":
        try:
            workers_per_gpu = int(args.workers_per_gpu)
            if workers_per_gpu < 1:
                parser.error("--workers-per-gpu must be >= 1")
        except ValueError:
            parser.error("--workers-per-gpu must be 'auto' or an integer")

    explicit_feature_options = _parse_bool_grid(args.explicit_feature_options)

    output_dir = Path(args.output_dir)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "command_line.txt").write_text(" ".join(sys.argv) + "\n")

    base_configs = generate_experiment_grid(
        dataset_paths=args.datasets,
        filter_max_lengths=args.max_lengths,
        model_names=args.model_names,
        chat_template_prompt=args.chat_template_prompt,
        filter_extractor_types=args.filter_extractor_types,
        learning_rates=args.learning_rates,
        epoch_counts=args.epoch_counts,
        include_explicit_feature_options=explicit_feature_options,
    )

    use_cache = args.cache or args.gpu_cache
    configs = []
    for base_config in base_configs:
        for repeat_idx in range(args.n_repeats):
            config = deepcopy(base_config)
            config.repeat_index = repeat_idx
            config.n_folds = args.n_folds
            config.flp_cache_hidden_states = use_cache
            config.hlm_cache_hidden_states = use_cache
            config.flp_downprojection_dim = None
            config.hlm_downprojection_dim = None
            config.use_explicit_confounders = config.use_explicit_features
            configs.append(config)
    random.Random(42).shuffle(configs)

    logger.info(
        "Generated %d base configs x %d repeats = %d experiments",
        len(base_configs),
        args.n_repeats,
        len(configs),
    )
    logger.info(
        "Mode: %s",
        "cached hidden states" if use_cache else "live LLM forward per batch",
    )

    completed_hashes = set()
    results_dict: Dict[str, Any] = {}
    if args.resume:
        results_dir = output_dir / "results"
        if results_dir.exists():
            for result_file in results_dir.glob("*.json"):
                completed_hashes.add(result_file.stem)
                with open(result_file) as f:
                    results_dict[result_file.stem] = json.load(f)
            logger.info("Resuming: found %d completed experiments", len(completed_hashes))

    pending_configs = [config for config in configs if config.config_hash() not in completed_hashes]
    if args.max_experiments:
        pending_configs = pending_configs[: args.max_experiments]

    print_grid_summary(
        pending_configs=pending_configs,
        completed_count=len(completed_hashes),
        n_repeats=args.n_repeats,
        base_config_count=len(base_configs),
    )

    if args.dry_run or not pending_configs:
        if not pending_configs:
            logger.info("No experiments to run")
        return

    cache_base_dir = str(output_dir / ".oci_cache")
    cache_groups = group_configs_by_cache_key(pending_configs, use_cache)

    if use_cache:
        wpg_per_device = {
            device: resolve_workers_per_gpu(args.workers_per_gpu, device, use_cache)
            for device in args.devices
        }
    else:
        wpg_per_device = {device: 1 for device in args.devices}

    progress_bar = tqdm(total=len(pending_configs), desc="X/W RF-CF experiments")

    for group_idx, (cache_hash, cache_info, group_configs) in enumerate(cache_groups):
        if not group_configs:
            continue

        real_cache_group = _is_real_cache_group(cache_hash, cache_info)
        if real_cache_group:
            logger.info("\n%s", "=" * 60)
            logger.info("Cache group %d/%d: %s", group_idx + 1, len(cache_groups), cache_hash)
            logger.info(
                "  max_length=%s, downprojection_dim=%s, dataset=%s",
                cache_info.get("max_length"),
                cache_info.get("downprojection_dim"),
                cache_info.get("dataset_name"),
            )
            logger.info("  %d experiment(s) in this group", len(group_configs))
            logger.info("%s", "=" * 60)

            cache = precompute_single_cache(
                cache_info,
                args.devices,
                cache_base_dir=cache_base_dir,
            )
            torch.set_default_dtype(torch.float32)
            cache.close()
            del cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            serializable_cache_info = {
                k: str(v) if isinstance(v, Path) else v
                for k, v in cache_info.items()
            }

            ctx = mp.get_context("spawn")
            job_queue_mp = ctx.Queue()
            progress_queue = ctx.Queue()
            for config in group_configs:
                job_queue_mp.put(config)

            processes = []
            for device in args.devices:
                for worker_idx in range(wpg_per_device[device]):
                    process = ctx.Process(
                        target=worker_process_fn,
                        args=(
                            device,
                            job_queue_mp,
                            progress_queue,
                            str(output_dir),
                            cache_hash,
                            serializable_cache_info,
                            args.gpu_cache,
                            cache_base_dir,
                        ),
                        name=f"worker-{device}-{worker_idx}",
                    )
                    process.start()
                    processes.append(process)

            completed_in_group = 0
            expected = len(group_configs)
            while completed_in_group < expected:
                alive = [process for process in processes if process.is_alive()]
                if not alive and completed_in_group < expected:
                    logger.error(
                        "All workers died with %d experiments remaining",
                        expected - completed_in_group,
                    )
                    break

                try:
                    _status, config_hash, result = progress_queue.get(timeout=5)
                    results_dict[config_hash] = result
                    completed_in_group += 1
                    progress_bar.update(1)
                    if result.get("skipped"):
                        progress_bar.set_postfix_str(
                            f"Skipped: {result.get('error', 'unknown')[:30]}"
                        )
                    else:
                        metrics = result.get("metrics", {})
                        progress_bar.set_postfix_str(
                            f"X/W RF-CF ITE corr: {metrics.get('ite_corr', float('nan')):.3f}"
                        )
                except Exception:
                    pass

            for process in processes:
                process.join(timeout=30)
                if process.is_alive():
                    logger.warning("Worker %s did not exit cleanly; terminating", process.name)
                    process.terminate()

        else:
            lock = threading.Lock()
            job_queue_t = queue.Queue()
            for config in group_configs:
                job_queue_t.put(config)

            threads = []
            for device in args.devices:
                n_threads = 1 if not use_cache else wpg_per_device[device]
                for worker_idx in range(n_threads):
                    thread = threading.Thread(
                        target=worker_thread,
                        args=(
                            device,
                            job_queue_t,
                            results_dict,
                            output_dir,
                            lock,
                            progress_bar,
                        ),
                        name=f"worker-{device}-{worker_idx}",
                    )
                    thread.start()
                    threads.append(thread)

            for thread in threads:
                thread.join()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    progress_bar.close()
    logger.info("Aggregating results...")
    aggregate_results(output_dir, results_dict)
    logger.info("Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
