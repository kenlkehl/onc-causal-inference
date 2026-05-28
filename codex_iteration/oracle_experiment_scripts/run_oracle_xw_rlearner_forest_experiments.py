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

from oci.config import ContrastiveEffectConfig, ExplicitFeatureSpec, TRAINABLE_EXTRACTOR_TYPES
from oci.data import (
    CachedHiddenStateDataset,
    ClinicalTextDataset,
    collate_batch,
    collate_cached_batch,
    prepare_cached_batch,
)
from oci.models.causal_text_forest import CausalTextForest
from oci.models.contrastive_causal_text_forest import ContrastiveCausalTextForest
from oci.models.gpu_hidden_state_store import GPUHiddenStateStore
from oci.models.hidden_state_cache import HiddenStateCache
from oci.models.causal_purity_hash_extractor import (
    build_token_hash_matrix,
    select_causal_purity_hashes,
)
from oci.training.contrastive_effect import (
    MatchedPairBatchSampler,
    PropensityBinBalancedBatchSampler,
    make_propensity_matched_pairs,
    make_propensity_bins,
)

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
    rlearner_mode: str = "staged_separate_nets"
    xw_feature_split: bool = True
    use_explicit_features: bool = False
    # Compatibility alias for older analysis scripts.
    use_explicit_confounders: bool = False

    feature_extractor_type: str = "frozen_llm_pooler"
    repeat_index: int = 0

    # Frozen LLM Pooler hyperparameters.
    flp_max_length: int = 50000
    flp_freeze_llm: bool = True
    flp_projection_dim: int = 128
    flp_gated_attention_dim: int = 128
    flp_downprojection_dim: Optional[int] = None
    flp_cache_hidden_states: bool = False
    flp_chat_template_prompt: Optional[str] = None
    flp_model_name: str = "Qwen/Qwen3.5-0.8B-Base"
    flp_dropout: float = 0.1
    flp_gradient_checkpointing: bool = True
    flp_attention_slots: int = 1
    flp_document_window: str = "tail"

    # Fixed training parameters.
    epochs: int = 30
    batch_size: int = 2
    learning_rate: float = 1e-4
    n_folds: int = 5
    rlearner_nuisance_folds: int = 5
    gamma_rlearner: float = 1.0
    effect_aux_outcome_weight: float = 0.0
    nuisance_potential_weight: float = 0.0
    effect_dr_weight: float = 0.0
    effect_dr_clip: float = 1.0
    effect_attention_entropy_weight: float = 0.0
    initialize_effect_from_nuisance: bool = False
    export_nuisance_potential_features: bool = False
    cf_export_shared_text_features: bool = True
    wx_nuisance_hidden_dim: int = 64
    wx_effect_hidden_dim: int = 64
    cf_forest_x_mode: str = "x_hidden_plus_tau"

    # Optional matched-contrastive X-stage replacement for per-patient R-loss.
    contrastive_effect_enabled: bool = False
    contrastive_bottleneck_dim: int = 8
    contrastive_hidden_dim: int = 64
    contrastive_batch_size: int = 16
    contrastive_n_propensity_bins: int = 10
    contrastive_overlap_min: float = 0.05
    contrastive_overlap_max: float = 0.95
    contrastive_min_arm_per_bin: int = 2
    contrastive_pairwise_matching: bool = False
    contrastive_pair_caliper: float = 0.05
    contrastive_lambda_factual: float = 1.0
    contrastive_lambda_contrast: float = 2.0
    contrastive_lambda_adversary: float = 0.05
    contrastive_lambda_pair_pull: float = 0.0
    contrastive_lambda_z_l2: float = 1e-4
    contrastive_target_clip: float = 1.0
    contrastive_forest_x_mode: str = "bottleneck_plus_tau"

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

    # Byte CNN hyperparameters.
    byte_embedding_dim: int = 32
    byte_conv_dim: int = 64
    byte_kernel_size: int = 7
    byte_num_conv_blocks: int = 4
    byte_chunk_size: int = 512
    byte_chunk_overlap: int = 64
    byte_max_chunks: int = 128
    byte_projection_dim: int = 128
    byte_dropout: float = 0.1

    # Outcome-guided token-ID hash screener parameters.
    purity_hash_top_k: int = 1
    purity_hash_alpha: float = 1.0
    purity_hash_beta: float = 2.0
    purity_hash_gamma_control: float = 1.0
    purity_hash_min_count: int = 20
    purity_hash_min_arm_count: int = 5
    purity_hash_candidate_k: int = 256
    purity_hash_student_hidden_dim: int = 64
    purity_hash_student_steps: int = 700
    purity_hash_student_lr: float = 0.03
    purity_hash_student_threshold: str = "threshold05"
    purity_hash_selector_steps: int = 800
    purity_hash_selector_lr: float = 1.0
    purity_hash_selector_temperature: float = 1.0
    purity_hash_selector_entropy: float = 0.0
    purity_hash_selector_threshold: str = "threshold05"

    _EXTRACTOR_PREFIXES = {
        "frozen_llm_pooler": {"flp_"},
        "frozen_llm_token_cnn": {"flp_"},
        "frozen_llm_stat_pooler": {"flp_"},
        "token_hash_embedding": {"flp_"},
        "causal_purity_hash": {"flp_", "purity_hash_"},
        "causal_purity_hash_student": {"flp_", "purity_hash_"},
        "neural_causal_hash_selector": {"flp_", "purity_hash_"},
        "hierarchical_llm": {"hlm_"},
        "hierarchical_cnn": {"hcnn_"},
        "hierarchical_gru": {"hgru_"},
        "simple_cnn": {"scnn_"},
        "byte_cnn": {"byte_"},
        "text_marker": set(),
    }
    _ALL_EXTRACTOR_PREFIXES = set().union(*_EXTRACTOR_PREFIXES.values())

    def __post_init__(self):
        self.model_type = "causal_forest"
        if self.contrastive_effect_enabled and self.contrastive_pairwise_matching:
            self.rlearner_mode = "pairwise_matched_contrastive_effect"
        elif self.contrastive_effect_enabled:
            self.rlearner_mode = "matched_contrastive_effect"
        else:
            self.rlearner_mode = "staged_separate_nets"
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
        # Tokenized non-cached text datasets were repeatedly failing in worker
        # processes/pin-memory threads on the oracle grid; keep this path in the
        # main process so trainable CNN/GRU probes can complete reliably.
        dl_kwargs = dict(num_workers=0)

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


def _oracle_contrastive_config(config: XWRLearnerForestConfig) -> ContrastiveEffectConfig:
    """Map oracle-runner flat fields to the library contrastive config."""
    return ContrastiveEffectConfig(
        enabled=config.contrastive_effect_enabled,
        bottleneck_dim=config.contrastive_bottleneck_dim,
        hidden_dim=config.contrastive_hidden_dim,
        batch_size=config.contrastive_batch_size,
        n_propensity_bins=config.contrastive_n_propensity_bins,
        overlap_min=config.contrastive_overlap_min,
        overlap_max=config.contrastive_overlap_max,
        min_arm_per_bin=config.contrastive_min_arm_per_bin,
        pairwise_matching=config.contrastive_pairwise_matching,
        pair_caliper=config.contrastive_pair_caliper,
        lambda_factual=config.contrastive_lambda_factual,
        lambda_contrast=config.contrastive_lambda_contrast,
        lambda_adversary=config.contrastive_lambda_adversary,
        lambda_pair_pull=config.contrastive_lambda_pair_pull,
        lambda_z_l2=config.contrastive_lambda_z_l2,
        target_clip=config.contrastive_target_clip,
        forest_x_mode=config.contrastive_forest_x_mode,
    )


def _make_xw_model(
    config: XWRLearnerForestConfig,
    device: torch.device,
    explicit_feature_specs: List[ExplicitFeatureSpec],
    gpu_store,
    hidden_state_cache,
    tokenizer_texts: Optional[List[str]] = None,
) -> CausalTextForest:
    """Create the staged X/W model with consistent oracle-runner settings."""
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
            wx_nuisance_hidden_dim=config.wx_nuisance_hidden_dim,
            wx_effect_hidden_dim=config.wx_effect_hidden_dim,
            dropout=0.2,
            cf_n_estimators=config.cf_n_estimators,
            cf_min_samples_leaf=config.cf_min_samples_leaf,
            cf_honest=True,
            cf_inference=True,
            cf_use_rlearner_representation=True,
            cf_gamma_rlearner=config.gamma_rlearner,
            cf_effect_aux_outcome_weight=config.effect_aux_outcome_weight,
            cf_nuisance_potential_weight=config.nuisance_potential_weight,
            cf_effect_dr_weight=config.effect_dr_weight,
            cf_effect_dr_clip=config.effect_dr_clip,
            cf_effect_attention_entropy_weight=config.effect_attention_entropy_weight,
            cf_export_nuisance_potential_features=config.export_nuisance_potential_features,
            cf_export_shared_text_features=config.cf_export_shared_text_features,
            cf_forest_x_mode=config.cf_forest_x_mode,
            explicit_feature_specs=explicit_feature_specs,
        )
    )

    model_class = ContrastiveCausalTextForest if config.contrastive_effect_enabled else CausalTextForest
    if config.contrastive_effect_enabled:
        model_kwargs["contrastive_effect_config"] = _oracle_contrastive_config(config)

    model = model_class(**model_kwargs)
    if config.feature_extractor_type in TRAINABLE_EXTRACTOR_TYPES and tokenizer_texts is not None:
        model.fit_tokenizer(tokenizer_texts)

    for name, param in model.named_parameters():
        if param.dtype != torch.float32:
            logger.warning(
                "Parameter %s has dtype %s; casting to float32",
                name,
                param.dtype,
            )
            param.data = param.data.float()
    return model


def _fit_explicit_feature_state(model: CausalTextForest, dataset) -> None:
    """Fit MLP and raw explicit-feature normalization from a training dataset."""
    if getattr(dataset, "explicit_feature_values", None):
        model.fit_explicit_features(dataset.explicit_feature_values)
        model.fit_explicit_feature_featurizer(dataset.explicit_feature_values)


def _train_nuisance_stage(
    model: CausalTextForest,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    config: XWRLearnerForestConfig,
    device: torch.device,
    use_cached: bool,
    gpu_store,
) -> None:
    """Train e(W), m(W); if val_loader is provided, restore the best val state."""
    params = [p for p in model.nuisance_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

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
            losses = model.train_nuisance_step(batch, alpha_propensity=1.0)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

        scheduler.step()

        if val_loader is None:
            continue

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch["treatment"] = batch["treatment"].to(device)
                batch["outcome"] = batch["outcome"].to(device)
                if use_cached:
                    prepare_cached_batch(batch, device, gpu_store=gpu_store)
                losses = model.train_nuisance_step(batch, alpha_propensity=1.0)
                val_loss += losses["loss"].item()

        val_loss /= max(len(val_loader), 1)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)


def _train_effect_stage(
    model: CausalTextForest,
    train_loader: DataLoader,
    nuisance_propensity: np.ndarray,
    nuisance_outcome: np.ndarray,
    nuisance_mu0: Optional[np.ndarray],
    nuisance_mu1: Optional[np.ndarray],
    config: XWRLearnerForestConfig,
    device: torch.device,
    use_cached: bool,
    gpu_store,
) -> None:
    """Train tau(X) from fixed outer-train OOF nuisance predictions."""
    if config.initialize_effect_from_nuisance:
        if not hasattr(model, "initialize_effect_from_nuisance"):
            raise AttributeError("Model does not support effect-from-nuisance initialization")
        model.initialize_effect_from_nuisance()

    use_contrastive = (
        config.contrastive_effect_enabled
        and hasattr(model, "train_effect_contrastive_step")
    )
    effect_loader = train_loader
    propensity_bin_ids = None
    use_pairwise_contrastive = use_contrastive and config.contrastive_pairwise_matching
    if use_contrastive:
        dataset_treatment = train_loader.dataset.treatments
        if hasattr(dataset_treatment, "detach"):
            dataset_treatment = dataset_treatment.detach().cpu().numpy()
        else:
            dataset_treatment = np.asarray(dataset_treatment)
        if use_pairwise_contrastive:
            matched_pairs = make_propensity_matched_pairs(
                propensity=nuisance_propensity,
                treatment=dataset_treatment,
                overlap_min=config.contrastive_overlap_min,
                overlap_max=config.contrastive_overlap_max,
                caliper=config.contrastive_pair_caliper,
                with_replacement=True,
            )
            sampler = MatchedPairBatchSampler(
                pairs=matched_pairs,
                batch_size=config.contrastive_batch_size,
                seed=42 + config.repeat_index,
            )
            logger.info(
                "Pairwise contrastive effect matches: %d treated/control pairs "
                "(caliper=%.3g)",
                len(matched_pairs),
                config.contrastive_pair_caliper,
            )
        else:
            propensity_bin_ids = make_propensity_bins(
                propensity=nuisance_propensity,
                treatment=dataset_treatment,
                n_bins=config.contrastive_n_propensity_bins,
                overlap_min=config.contrastive_overlap_min,
                overlap_max=config.contrastive_overlap_max,
                min_arm_per_bin=config.contrastive_min_arm_per_bin,
            )
            sampler = PropensityBinBalancedBatchSampler(
                treatment=dataset_treatment,
                bin_ids=propensity_bin_ids,
                batch_size=config.contrastive_batch_size,
                min_arm_per_bin=config.contrastive_min_arm_per_bin,
                seed=42 + config.repeat_index,
            )
        loader_kwargs = {}
        if getattr(train_loader, "num_workers", 0) > 0:
            loader_kwargs["num_workers"] = train_loader.num_workers
            loader_kwargs["persistent_workers"] = getattr(train_loader, "persistent_workers", False)
            loader_kwargs["pin_memory"] = getattr(train_loader, "pin_memory", False)
        effect_loader = DataLoader(
            train_loader.dataset,
            batch_sampler=sampler,
            collate_fn=train_loader.collate_fn,
            **loader_kwargs,
        )
        if propensity_bin_ids is not None:
            logger.info(
                "Contrastive effect bins: %d bins, %d/%d samples in overlap",
                len(np.unique(propensity_bin_ids[propensity_bin_ids >= 0])),
                int(np.sum(propensity_bin_ids >= 0)),
                len(propensity_bin_ids),
            )

    params = [p for p in model.effect_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    for _epoch in range(config.epochs):
        model.train()
        for batch in effect_loader:
            batch["treatment"] = batch["treatment"].to(device)
            batch["outcome"] = batch["outcome"].to(device)
            if use_cached:
                prepare_cached_batch(batch, device, gpu_store=gpu_store)

            batch_ids = np.asarray(batch["text_id"], dtype=int)
            e_hat = torch.as_tensor(
                nuisance_propensity[batch_ids],
                dtype=torch.float32,
                device=device,
            )
            m_hat = torch.as_tensor(
                nuisance_outcome[batch_ids],
                dtype=torch.float32,
                device=device,
            )
            mu0_hat = None
            mu1_hat = None
            if nuisance_mu0 is not None and nuisance_mu1 is not None:
                mu0_hat = torch.as_tensor(
                    nuisance_mu0[batch_ids],
                    dtype=torch.float32,
                    device=device,
                )
                mu1_hat = torch.as_tensor(
                    nuisance_mu1[batch_ids],
                    dtype=torch.float32,
                    device=device,
                )

            optimizer.zero_grad()
            if use_contrastive:
                if use_pairwise_contrastive:
                    losses = model.train_effect_pairwise_contrastive_step(
                        batch,
                        e_hat=e_hat,
                        m_hat=m_hat,
                    )
                else:
                    bin_ids = torch.as_tensor(
                        propensity_bin_ids[batch_ids],
                        dtype=torch.long,
                        device=device,
                    )
                    losses = model.train_effect_contrastive_step(
                        batch,
                        e_hat=e_hat,
                        m_hat=m_hat,
                        bin_ids=bin_ids,
                    )
            else:
                losses = model.train_effect_r_step(
                    batch,
                    e_hat=e_hat,
                    m_hat=m_hat,
                    gamma_rlearner=config.gamma_rlearner,
                    mu0_hat=mu0_hat,
                    mu1_hat=mu1_hat,
                )
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

        scheduler.step()


def _fit_predict_hash_nuisance(
    W_train: np.ndarray,
    W_test: np.ndarray,
    treatment_train: np.ndarray,
    outcome_train: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit generic nuisance models on hash-SVD controls for reporting metrics."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    propensity_model = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=20,
        random_state=seed,
        n_jobs=-1,
    )
    propensity_model.fit(W_train, treatment_train)
    pred_propensity = propensity_model.predict_proba(W_test)[:, 1]

    pred_y = []
    for arm in (0, 1):
        mask = treatment_train == arm
        if mask.sum() == 0:
            pred_y.append(np.full(W_test.shape[0], float(outcome_train.mean())))
            continue
        outcome_model = RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=20,
            random_state=seed + 100 + arm,
            n_jobs=-1,
        )
        outcome_model.fit(W_train[mask], outcome_train[mask])
        pred_y.append(np.clip(outcome_model.predict(W_test), 1e-3, 1 - 1e-3))

    return (
        np.clip(pred_propensity, 1e-3, 1 - 1e-3),
        pred_y[0],
        pred_y[1],
    )


def _train_hash_teacher_student(
    X_train,
    teacher_train: np.ndarray,
    X_test,
    config: XWRLearnerForestConfig,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Train a small neural student to predict a train-fold anonymous hash teacher."""
    from sklearn.metrics import roc_auc_score

    torch.manual_seed(seed)
    X_train_dense = torch.as_tensor(X_train.toarray(), dtype=torch.float32)
    y_train = torch.as_tensor(teacher_train.astype(np.float32)).view(-1, 1)
    model = torch.nn.Sequential(
        torch.nn.Linear(X_train_dense.shape[1], config.purity_hash_student_hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(config.purity_hash_student_hidden_dim, 1),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.purity_hash_student_lr,
        weight_decay=5e-5,
    )
    positives = max(float(y_train.sum()), 1.0)
    negatives = max(float(len(y_train) - y_train.sum()), 1.0)
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negatives / positives], dtype=torch.float32)
    )

    best_state = None
    best_loss = float("inf")
    for _step in range(config.purity_hash_student_steps):
        optimizer.zero_grad()
        loss = loss_fn(model(X_train_dense), y_train)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach())
        if loss_value < best_loss:
            best_loss = loss_value
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    with torch.no_grad():
        train_score = torch.sigmoid(model(X_train_dense)).numpy().reshape(-1)
        test_score = torch.sigmoid(
            model(torch.as_tensor(X_test.toarray(), dtype=torch.float32))
        ).numpy().reshape(-1)

    train_feature = train_score
    test_feature = test_score
    if config.purity_hash_student_threshold == "threshold05":
        train_feature = (train_score >= 0.5).astype(np.float32)
        test_feature = (test_score >= 0.5).astype(np.float32)
    elif config.purity_hash_student_threshold == "train_freq":
        threshold = float(np.quantile(train_score, 1.0 - float(teacher_train.mean())))
        train_feature = (train_score >= threshold).astype(np.float32)
        test_feature = (test_score >= threshold).astype(np.float32)
    elif config.purity_hash_student_threshold != "none":
        raise ValueError(
            "purity_hash_student_threshold must be one of: none, threshold05, train_freq"
        )

    diagnostics: Dict[str, float] = {"student_loss": best_loss}
    if len(np.unique(teacher_train)) > 1:
        diagnostics["student_train_auc"] = float(roc_auc_score(teacher_train, train_score))
        diagnostics["student_train_corr"] = float(
            np.corrcoef(train_score, teacher_train)[0, 1]
        )
    return train_feature.reshape(-1, 1), test_feature.reshape(-1, 1), diagnostics


def _train_neural_hash_selector(
    X_train_hash,
    X_test_hash,
    original_hashes: np.ndarray,
    treatment_train: np.ndarray,
    outcome_train: np.ndarray,
    config: XWRLearnerForestConfig,
    device: torch.device,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Train a differentiable selector over anonymous token-hash columns."""
    if device.type == "cuda" and torch.cuda.is_available():
        selector_device = device
    elif torch.cuda.is_available():
        selector_device = torch.device("cuda:0")
    else:
        selector_device = torch.device("cpu")

    treatment_train = np.asarray(treatment_train, dtype=int)
    outcome_train = np.asarray(outcome_train, dtype=float)
    treated_mask = treatment_train == 1
    control_mask = ~treated_mask
    present_count = np.asarray(X_train_hash.sum(axis=0)).ravel()
    present_treated = np.asarray(X_train_hash[treated_mask].sum(axis=0)).ravel()
    present_control = np.asarray(X_train_hash[control_mask].sum(axis=0)).ravel()
    min_arm = np.minimum.reduce([
        present_treated,
        present_control,
        int(treated_mask.sum()) - present_treated,
        int(control_mask.sum()) - present_control,
    ])
    candidates = np.flatnonzero(
        (present_count >= config.purity_hash_min_count)
        & (present_count <= len(treatment_train) - config.purity_hash_min_count)
        & (min_arm >= config.purity_hash_min_arm_count)
    )
    if candidates.size == 0:
        raise RuntimeError(
            "No neural hash selector candidates passed count/overlap thresholds; "
            "relax purity_hash_min_count or purity_hash_min_arm_count."
        )

    X_train_dense = torch.as_tensor(
        X_train_hash[:, candidates].toarray(),
        dtype=torch.float32,
        device=selector_device,
    )
    treatment_tensor = torch.as_tensor(
        treatment_train.astype(np.float32),
        dtype=torch.float32,
        device=selector_device,
    )
    outcome_tensor = torch.as_tensor(
        outcome_train.astype(np.float32),
        dtype=torch.float32,
        device=selector_device,
    )

    torch.manual_seed(seed)
    logits = torch.nn.Parameter(
        torch.randn(len(candidates), dtype=torch.float32, device=selector_device) * 0.01
    )
    optimizer = torch.optim.Adam([logits], lr=config.purity_hash_selector_lr)
    treated = treatment_tensor
    control = 1.0 - treatment_tensor
    eps = 1e-4
    best_state: Optional[Dict[str, Any]] = None
    best_score = -float("inf")

    temperature = max(float(config.purity_hash_selector_temperature), 1e-3)
    for _step in range(config.purity_hash_selector_steps):
        optimizer.zero_grad()
        probabilities = torch.softmax(logits / temperature, dim=0)
        selected_feature = X_train_dense.mv(probabilities)

        def weighted_mean(mask: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            return (
                (mask * weights * outcome_tensor).sum()
                / (mask * weights).sum().clamp_min(eps)
            )

        mean_treated_present = weighted_mean(treated, selected_feature)
        mean_control_present = weighted_mean(control, selected_feature)
        mean_treated_absent = weighted_mean(treated, 1.0 - selected_feature)
        mean_control_absent = weighted_mean(control, 1.0 - selected_feature)
        te_diff = (
            mean_treated_present
            - mean_control_present
            - (mean_treated_absent - mean_control_absent)
        )
        control_diff = mean_control_present - mean_control_absent
        frequency = selected_feature.mean()
        soft_min_arm = torch.minimum(
            torch.minimum((treated * selected_feature).sum(), (control * selected_feature).sum()),
            torch.minimum(
                (treated * (1.0 - selected_feature)).sum(),
                (control * (1.0 - selected_feature)).sum(),
            ),
        ) / len(treatment_train)
        score = (
            (te_diff - config.purity_hash_gamma_control * torch.abs(control_diff))
            * torch.clamp(soft_min_arm, min=0.0)
            * torch.clamp(1.0 - frequency, min=0.0).pow(config.purity_hash_beta)
        )
        entropy = -(probabilities * torch.log(probabilities + 1e-12)).sum()
        loss = -score + config.purity_hash_selector_entropy * entropy
        loss.backward()
        optimizer.step()

        score_value = float(score.detach().cpu())
        if score_value > best_score:
            top_local = int(torch.argmax(probabilities).detach().cpu())
            best_score = score_value
            best_state = {
                "probabilities": probabilities.detach().cpu().numpy(),
                "top_local": top_local,
                "score": score_value,
                "te_diff": float(te_diff.detach().cpu()),
                "control_diff": float(control_diff.detach().cpu()),
                "frequency": float(frequency.detach().cpu()),
                "top_probability": float(probabilities[top_local].detach().cpu()),
                "entropy": float(entropy.detach().cpu()),
            }

    if best_state is None:
        raise RuntimeError("Neural hash selector failed to record a best state")

    probabilities = best_state["probabilities"]
    train_score = X_train_hash[:, candidates].dot(probabilities).astype(np.float32)
    test_score = X_test_hash[:, candidates].dot(probabilities).astype(np.float32)
    train_feature = train_score
    test_feature = test_score
    if config.purity_hash_selector_threshold == "threshold05":
        train_feature = (train_score >= 0.5).astype(np.float32)
        test_feature = (test_score >= 0.5).astype(np.float32)
    elif config.purity_hash_selector_threshold == "argmax":
        top_column = candidates[best_state["top_local"]]
        train_feature = X_train_hash[:, top_column].toarray().ravel().astype(np.float32)
        test_feature = X_test_hash[:, top_column].toarray().ravel().astype(np.float32)
    elif config.purity_hash_selector_threshold != "none":
        raise ValueError(
            "purity_hash_selector_threshold must be one of: none, threshold05, argmax"
        )

    top_column = int(candidates[best_state["top_local"]])
    diagnostics = {
        "selected_hashes": [int(original_hashes[top_column])],
        "scores": [best_state["score"]],
        "te_diff": [best_state["te_diff"]],
        "control_diff": [best_state["control_diff"]],
        "counts": [float(present_count[top_column])],
        "selector_candidate_count": int(len(candidates)),
        "selector_top_probability": best_state["top_probability"],
        "selector_entropy": best_state["entropy"],
        "selector_frequency": best_state["frequency"],
        "selector_device": str(selector_device),
    }
    return train_feature.reshape(-1, 1), test_feature.reshape(-1, 1), diagnostics


def _run_causal_purity_hash_experiment(
    config: XWRLearnerForestConfig,
    df: pd.DataFrame,
    device: torch.device,
) -> Dict[str, Any]:
    """Run the no-string causal-purity hash screener baseline."""
    from econml.dml import CausalForestDML
    from sklearn.decomposition import TruncatedSVD
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    text_column = "clinical_text"
    df = df.reset_index(drop=True)
    logger.info(
        "Building token-ID hash matrix with tokenizer=%s, max_length=%s, window=%s",
        config.flp_model_name,
        config.flp_max_length,
        config.flp_document_window,
    )
    X_hash, original_hashes = build_token_hash_matrix(
        df[text_column].tolist(),
        model_name=config.flp_model_name,
        max_length=config.flp_max_length,
        document_window=config.flp_document_window,
        min_count=1,
    )
    logger.info(
        "Token-ID hash matrix shape=%s, selecting top_k=%d by train-fold causal purity",
        X_hash.shape,
        (
            config.purity_hash_candidate_k
            if config.feature_extractor_type == "causal_purity_hash_student"
            else X_hash.shape[1]
            if config.feature_extractor_type == "neural_causal_hash_selector"
            else config.purity_hash_top_k
        ),
    )

    kf = KFold(n_splits=config.n_folds, shuffle=True, random_state=42 + config.repeat_index)
    all_predictions = []
    diagnostics = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        treatment_train = train_df["treatment_indicator"].to_numpy(dtype=int)
        outcome_train = train_df["outcome_indicator"].to_numpy(dtype=float)

        is_student = config.feature_extractor_type == "causal_purity_hash_student"
        is_neural_selector = config.feature_extractor_type == "neural_causal_hash_selector"
        student_diag: Dict[str, Any] = {}
        if is_neural_selector:
            X_train, X_test, student_diag = _train_neural_hash_selector(
                X_hash[train_idx],
                X_hash[test_idx],
                original_hashes=original_hashes,
                treatment_train=treatment_train,
                outcome_train=outcome_train,
                config=config,
                device=device,
                seed=7000 + config.repeat_index * 100 + fold,
            )
            X_train = X_train.astype(np.float32)
            X_test = X_test.astype(np.float32)
            selection = None
        else:
            selection_top_k = (
                config.purity_hash_candidate_k
                if is_student
                else config.purity_hash_top_k
            )
            selection = select_causal_purity_hashes(
                X_hash[train_idx],
                original_hashes=original_hashes,
                treatment=treatment_train,
                outcome=outcome_train,
                top_k=selection_top_k,
                alpha=config.purity_hash_alpha,
                beta=config.purity_hash_beta,
                gamma_control=config.purity_hash_gamma_control,
                min_count=config.purity_hash_min_count,
                min_arm_count=config.purity_hash_min_arm_count,
            )
            if len(selection.columns) == 0 or not np.isfinite(selection.scores[0]):
                raise RuntimeError(
                    "No finite causal-purity hash features selected for fold "
                    f"{fold + 1}; relax purity_hash_* thresholds."
                )

        if is_student and selection is not None:
            teacher_column = selection.columns[0]
            teacher_train = X_hash[train_idx][:, teacher_column].toarray().reshape(-1)
            X_candidates_train = X_hash[train_idx][:, selection.columns]
            X_candidates_test = X_hash[test_idx][:, selection.columns]
            X_train, X_test, student_diag = _train_hash_teacher_student(
                X_candidates_train,
                teacher_train=teacher_train,
                X_test=X_candidates_test,
                config=config,
                seed=5000 + config.repeat_index * 100 + fold,
            )
            X_train = X_train.astype(np.float32)
            X_test = X_test.astype(np.float32)
        elif not is_neural_selector and selection is not None:
            X_train = X_hash[train_idx][:, selection.columns].toarray().astype(np.float32)
            X_test = X_hash[test_idx][:, selection.columns].toarray().astype(np.float32)

        n_components = min(64, len(train_idx) - 2, X_hash.shape[1] - 1)
        if n_components >= 1:
            svd = TruncatedSVD(
                n_components=n_components,
                random_state=1000 + config.repeat_index * 100 + fold,
            )
            W_train = svd.fit_transform(X_hash[train_idx]).astype(np.float32)
            W_test = svd.transform(X_hash[test_idx]).astype(np.float32)
        else:
            W_train = X_train
            W_test = X_test

        cf_seed = 3000 + config.repeat_index * 100 + fold
        cf_model = CausalForestDML(
            model_y=RandomForestRegressor(
                n_estimators=200,
                min_samples_leaf=20,
                random_state=cf_seed + 1,
                n_jobs=-1,
            ),
            model_t=RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=20,
                random_state=cf_seed + 2,
                n_jobs=-1,
            ),
            discrete_treatment=True,
            n_estimators=max(400, config.cf_n_estimators),
            min_samples_leaf=max(10, config.cf_min_samples_leaf),
            random_state=cf_seed + 3,
            n_jobs=-1,
            cv=3,
        )
        cf_model.fit(Y=outcome_train, T=treatment_train, X=X_train, W=W_train)
        pred_tau = cf_model.effect(X_test).reshape(-1)

        tau_lower = None
        tau_upper = None
        try:
            tau_lower, tau_upper = cf_model.effect_interval(X_test)
            tau_lower = np.asarray(tau_lower).reshape(-1)
            tau_upper = np.asarray(tau_upper).reshape(-1)
        except Exception as exc:
            logger.debug("Causal-purity hash CI unavailable on fold %d: %s", fold + 1, exc)

        pred_propensity, pred_y0, pred_y1 = _fit_predict_hash_nuisance(
            W_train=W_train,
            W_test=W_test,
            treatment_train=treatment_train,
            outcome_train=outcome_train,
            seed=4000 + config.repeat_index * 100 + fold,
        )

        if selection is None:
            fold_diag = {"fold": fold + 1}
        else:
            fold_diag = {
                "fold": fold + 1,
                "selected_hashes": selection.original_hashes.astype(int).tolist(),
                "scores": selection.scores.astype(float).tolist(),
                "te_diff": selection.te_diff.astype(float).tolist(),
                "control_diff": selection.control_diff.astype(float).tolist(),
                "counts": selection.counts.astype(float).tolist(),
            }
        fold_diag.update(student_diag)
        diagnostics.append(fold_diag)
        logged_hashes = fold_diag["selected_hashes"][: min(5, len(fold_diag["selected_hashes"]))]
        logger.info(
            "Fold %d causal-purity hashes=%s%s scores=%s",
            fold + 1,
            logged_hashes,
            "..." if len(fold_diag["selected_hashes"]) > len(logged_hashes) else "",
            [round(v, 4) for v in fold_diag["scores"][: len(logged_hashes)]],
        )

        fold_preds = test_df.copy()
        fold_preds["pred_y0_prob"] = pred_y0
        fold_preds["pred_y1_prob"] = pred_y1
        fold_preds["pred_ite_prob"] = pred_tau
        fold_preds["pred_propensity"] = pred_propensity
        fold_preds["pred_tau"] = pred_tau
        fold_preds["cv_fold"] = fold + 1
        if tau_lower is not None and tau_upper is not None:
            fold_preds["pred_tau_lower"] = tau_lower
            fold_preds["pred_tau_upper"] = tau_upper

        all_predictions.append(fold_preds)

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
    return {"metrics": metrics, "n_samples": len(results_df), "diagnostics": diagnostics}


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
    if config.feature_extractor_type in {
        "causal_purity_hash",
        "causal_purity_hash_student",
        "neural_causal_hash_selector",
    }:
        return _run_causal_purity_hash_experiment(config, df, device)

    text_column = "clinical_text"
    batch_size = config.batch_size
    df = df.reset_index(drop=True)
    kf = KFold(n_splits=config.n_folds, shuffle=True, random_state=42 + config.repeat_index)

    all_predictions = []
    use_cached = gpu_store is not None or hidden_state_cache is not None

    for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        n_inner = min(config.rlearner_nuisance_folds, len(train_df))
        if n_inner < 2:
            raise ValueError("rlearner_nuisance_folds requires at least 2 outer-train samples")
        inner_kf = KFold(
            n_splits=n_inner,
            shuffle=True,
            random_state=10_000 + 42 + config.repeat_index + fold,
        )
        oof_propensity = np.full(len(train_df), np.nan, dtype=np.float32)
        oof_outcome = np.full(len(train_df), np.nan, dtype=np.float32)
        oof_mu0 = np.full(len(train_df), np.nan, dtype=np.float32)
        oof_mu1 = np.full(len(train_df), np.nan, dtype=np.float32)

        for inner_train_pos, inner_val_pos in inner_kf.split(train_df):
            inner_train_df = train_df.iloc[inner_train_pos]
            inner_val_df = train_df.iloc[inner_val_pos]
            inner_train_idx = np.asarray(train_idx)[inner_train_pos]
            inner_val_idx = np.asarray(train_idx)[inner_val_pos]

            inner_model = _make_xw_model(
                config,
                device,
                explicit_feature_specs,
                gpu_store,
                hidden_state_cache,
                tokenizer_texts=inner_train_df[text_column].tolist(),
            )
            (
                inner_train_dataset,
                _inner_val_dataset,
                inner_train_loader,
                inner_val_loader,
                _inner_collate_fn,
                _inner_dl_kwargs,
            ) = _create_datasets_and_loaders(
                inner_train_df,
                inner_val_df,
                inner_train_idx,
                inner_val_idx,
                text_column,
                explicit_feature_cols,
                batch_size,
                hidden_state_cache,
                gpu_store,
            )
            _fit_explicit_feature_state(inner_model, inner_train_dataset)
            _train_nuisance_stage(
                inner_model,
                inner_train_loader,
                inner_val_loader,
                config,
                device,
                use_cached,
                gpu_store,
            )
            cf_kwargs = dict(gpu_store=gpu_store) if use_cached else {}
            if config.effect_dr_weight > 0:
                nuisance_components = inner_model.predict_nuisance_components(
                    inner_val_loader,
                    **cf_kwargs,
                )
                prop_hat = nuisance_components["propensity"]
                outcome_hat = nuisance_components["outcome"]
                oof_mu0[inner_val_pos] = nuisance_components["mu0"]
                oof_mu1[inner_val_pos] = nuisance_components["mu1"]
            else:
                prop_hat, outcome_hat = inner_model.predict_nuisance(
                    inner_val_loader,
                    **cf_kwargs,
                )
            oof_propensity[inner_val_pos] = prop_hat
            oof_outcome[inner_val_pos] = outcome_hat

            del inner_model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if np.isnan(oof_propensity).any() or np.isnan(oof_outcome).any():
            raise RuntimeError("Incomplete out-of-fold nuisance predictions")
        use_dr_pseudo_outcomes = config.effect_dr_weight > 0
        if use_dr_pseudo_outcomes and (np.isnan(oof_mu0).any() or np.isnan(oof_mu1).any()):
            raise RuntimeError("Incomplete out-of-fold potential-outcome nuisance predictions")

        model = _make_xw_model(
            config,
            device,
            explicit_feature_specs,
            gpu_store,
            hidden_state_cache,
            tokenizer_texts=train_df[text_column].tolist(),
        )
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

        _fit_explicit_feature_state(model, train_dataset)
        _train_nuisance_stage(
            model,
            train_loader,
            None,
            config,
            device,
            use_cached,
            gpu_store,
        )
        _train_effect_stage(
            model,
            train_loader,
            oof_propensity,
            oof_outcome,
            oof_mu0 if use_dr_pseudo_outcomes else None,
            oof_mu1 if use_dr_pseudo_outcomes else None,
            config,
            device,
            use_cached,
            gpu_store,
        )

        train_eval_loader = _make_combined_loader(
            train_df,
            np.asarray(train_idx),
            text_column,
            explicit_feature_cols,
            batch_size,
            hidden_state_cache,
            gpu_store,
            dl_kwargs,
        )

        train_T = train_df["treatment_indicator"].values
        train_Y = train_df["outcome_indicator"].values
        cf_kwargs = dict(gpu_store=gpu_store) if use_cached else {}
        model.train_causal_forest(train_eval_loader, train_T, train_Y, **cf_kwargs)
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

    output = {
        "config": asdict(config),
        "metrics": result["metrics"],
        "n_samples": result["n_samples"],
        "skipped": False,
        "error": None,
    }
    if "diagnostics" in result:
        output["diagnostics"] = result["diagnostics"]
    return output


def generate_experiment_grid(
    dataset_paths: List[str],
    filter_max_lengths: Optional[List[int]] = None,
    model_names: Optional[List[str]] = None,
    chat_template_prompt: Optional[str] = None,
    filter_extractor_types: Optional[List[str]] = None,
    learning_rates: Optional[List[float]] = None,
    epoch_counts: Optional[List[int]] = None,
    include_explicit_feature_options: Optional[List[bool]] = None,
    contrastive_effect_enabled: bool = False,
    contrastive_bottleneck_dim: int = 8,
    contrastive_hidden_dim: int = 64,
    contrastive_batch_size: int = 16,
    contrastive_n_propensity_bins: int = 10,
    contrastive_overlap_min: float = 0.05,
    contrastive_overlap_max: float = 0.95,
    contrastive_min_arm_per_bin: int = 2,
    contrastive_pairwise_matching: bool = False,
    contrastive_pair_caliper: float = 0.05,
    contrastive_lambda_factual: float = 1.0,
    contrastive_lambda_contrast: float = 2.0,
    contrastive_lambda_adversary: float = 0.05,
    contrastive_lambda_pair_pull: float = 0.0,
    contrastive_lambda_z_l2: float = 1e-4,
    contrastive_target_clip: float = 1.0,
    contrastive_forest_x_mode: str = "bottleneck_plus_tau",
    wx_nuisance_hidden_dims: Optional[List[int]] = None,
    wx_effect_hidden_dims: Optional[List[int]] = None,
    cf_forest_x_modes: Optional[List[str]] = None,
    cf_export_shared_text_feature_options: Optional[List[bool]] = None,
    byte_max_chunks_options: Optional[List[int]] = None,
    flp_attention_slot_options: Optional[List[int]] = None,
    flp_document_window_options: Optional[List[str]] = None,
    purity_hash_top_k: int = 1,
    purity_hash_alpha: float = 1.0,
    purity_hash_beta: float = 2.0,
    purity_hash_gamma_control: float = 1.0,
    purity_hash_min_count: int = 20,
    purity_hash_min_arm_count: int = 5,
    purity_hash_candidate_k: int = 256,
    purity_hash_student_hidden_dim: int = 64,
    purity_hash_student_steps: int = 700,
    purity_hash_student_lr: float = 0.03,
    purity_hash_student_threshold: str = "threshold05",
    purity_hash_selector_steps: int = 800,
    purity_hash_selector_lr: float = 1.0,
    purity_hash_selector_temperature: float = 1.0,
    purity_hash_selector_entropy: float = 0.0,
    purity_hash_selector_threshold: str = "threshold05",
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
        epoch_counts = [50]
    if include_explicit_feature_options is None:
        include_explicit_feature_options = [False, True]
    if wx_nuisance_hidden_dims is None:
        wx_nuisance_hidden_dims = [64]
    if wx_effect_hidden_dims is None:
        wx_effect_hidden_dims = [64]
    if cf_forest_x_modes is None:
        cf_forest_x_modes = ["x_hidden_plus_tau"]
    if cf_export_shared_text_feature_options is None:
        cf_export_shared_text_feature_options = [True]

    datasets = [(p, Path(p).name) for p in dataset_paths]
    all_extractor_types = [
        "frozen_llm_pooler",
        "frozen_llm_token_cnn",
        "frozen_llm_stat_pooler",
        "token_hash_embedding",
        "causal_purity_hash",
        "causal_purity_hash_student",
        "neural_causal_hash_selector",
        "hierarchical_llm",
        "hierarchical_cnn",
        "hierarchical_gru",
        "simple_cnn",
        "byte_cnn",
        "text_marker",
    ]
    default_extractor_types = [e for e in all_extractor_types if e != "text_marker"]
    extractor_types = default_extractor_types
    if filter_extractor_types:
        extractor_types = [e for e in all_extractor_types if e in filter_extractor_types]

    configs: List[XWRLearnerForestConfig] = []

    for ext_type in extractor_types:
        if ext_type in {
            "frozen_llm_pooler",
            "frozen_llm_token_cnn",
            "frozen_llm_stat_pooler",
            "token_hash_embedding",
            "causal_purity_hash",
            "causal_purity_hash_student",
            "neural_causal_hash_selector",
        }:
            max_lengths = filter_max_lengths or [50000]
            attention_slot_options = flp_attention_slot_options or [1]
            document_window_options = flp_document_window_options or ["tail"]
            chat_template_options = [None]
            if chat_template_prompt is not None:
                chat_template_options = [None, chat_template_prompt]

            for (
                dataset_path,
                dataset_name,
            ), max_len, slots, doc_window, use_feats, ctp, mn, lr, ep in itertools.product(
                datasets,
                max_lengths,
                attention_slot_options,
                document_window_options,
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
                        feature_extractor_type=ext_type,
                        flp_max_length=max_len,
                        flp_downprojection_dim=None,
                        flp_attention_slots=slots,
                        flp_document_window=doc_window,
                        flp_model_name=mn,
                        flp_chat_template_prompt=ctp,
                        learning_rate=lr,
                        epochs=ep,
                    )
                )

        elif ext_type == "hierarchical_llm":
            chunk_size = 2048
            chunk_overlap = 256
            max_chunks_options = [16]

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

        elif ext_type == "byte_cnn":
            max_chunks_options = byte_max_chunks_options or [128]
            for (
                dataset_path,
                dataset_name,
            ), use_feats, max_chunks, lr, ep in itertools.product(
                datasets,
                include_explicit_feature_options,
                max_chunks_options,
                learning_rates,
                epoch_counts,
            ):
                configs.append(
                    XWRLearnerForestConfig(
                        dataset_path=dataset_path,
                        dataset_name=dataset_name,
                        use_explicit_features=use_feats,
                        feature_extractor_type="byte_cnn",
                        byte_max_chunks=max_chunks,
                        learning_rate=lr,
                        epochs=ep,
                    )
                )

        elif ext_type == "text_marker":
            for (
                dataset_path,
                dataset_name,
            ), use_feats, lr, ep in itertools.product(
                datasets,
                include_explicit_feature_options,
                learning_rates,
                epoch_counts,
            ):
                configs.append(
                    XWRLearnerForestConfig(
                        dataset_path=dataset_path,
                        dataset_name=dataset_name,
                        use_explicit_features=use_feats,
                        feature_extractor_type="text_marker",
                        learning_rate=lr,
                        epochs=ep,
                    )
                )

    expanded_configs: List[XWRLearnerForestConfig] = []
    for cfg, w_dim, x_dim, x_mode, export_shared in itertools.product(
        configs,
        wx_nuisance_hidden_dims,
        wx_effect_hidden_dims,
        cf_forest_x_modes,
        cf_export_shared_text_feature_options,
    ):
        expanded_cfg = deepcopy(cfg)
        expanded_cfg.wx_nuisance_hidden_dim = w_dim
        expanded_cfg.wx_effect_hidden_dim = x_dim
        expanded_cfg.cf_forest_x_mode = x_mode
        expanded_cfg.cf_export_shared_text_features = export_shared
        expanded_configs.append(expanded_cfg)
    configs = expanded_configs

    for cfg in configs:
        cfg.contrastive_effect_enabled = contrastive_effect_enabled
        cfg.contrastive_bottleneck_dim = contrastive_bottleneck_dim
        cfg.contrastive_hidden_dim = contrastive_hidden_dim
        cfg.contrastive_batch_size = contrastive_batch_size
        cfg.contrastive_n_propensity_bins = contrastive_n_propensity_bins
        cfg.contrastive_overlap_min = contrastive_overlap_min
        cfg.contrastive_overlap_max = contrastive_overlap_max
        cfg.contrastive_min_arm_per_bin = contrastive_min_arm_per_bin
        cfg.contrastive_pairwise_matching = contrastive_pairwise_matching
        cfg.contrastive_pair_caliper = contrastive_pair_caliper
        cfg.contrastive_lambda_factual = contrastive_lambda_factual
        cfg.contrastive_lambda_contrast = contrastive_lambda_contrast
        cfg.contrastive_lambda_adversary = contrastive_lambda_adversary
        cfg.contrastive_lambda_pair_pull = contrastive_lambda_pair_pull
        cfg.contrastive_lambda_z_l2 = contrastive_lambda_z_l2
        cfg.contrastive_target_clip = contrastive_target_clip
        cfg.contrastive_forest_x_mode = contrastive_forest_x_mode
        cfg.purity_hash_top_k = purity_hash_top_k
        cfg.purity_hash_alpha = purity_hash_alpha
        cfg.purity_hash_beta = purity_hash_beta
        cfg.purity_hash_gamma_control = purity_hash_gamma_control
        cfg.purity_hash_min_count = purity_hash_min_count
        cfg.purity_hash_min_arm_count = purity_hash_min_arm_count
        cfg.purity_hash_candidate_k = purity_hash_candidate_k
        cfg.purity_hash_student_hidden_dim = purity_hash_student_hidden_dim
        cfg.purity_hash_student_steps = purity_hash_student_steps
        cfg.purity_hash_student_lr = purity_hash_student_lr
        cfg.purity_hash_student_threshold = purity_hash_student_threshold
        cfg.purity_hash_selector_steps = purity_hash_selector_steps
        cfg.purity_hash_selector_lr = purity_hash_selector_lr
        cfg.purity_hash_selector_temperature = purity_hash_selector_temperature
        cfg.purity_hash_selector_entropy = purity_hash_selector_entropy
        cfg.purity_hash_selector_threshold = purity_hash_selector_threshold
        cfg.__post_init__()

    random.Random(42).shuffle(configs)
    return configs


def _is_real_cache_group(cache_hash: str, cache_info: Optional[dict]) -> bool:
    return bool(cache_info) and not cache_hash.startswith("__no_cache__")


def randomize_execution_groups(
    cache_groups: List[Tuple[str, dict, List[XWRLearnerForestConfig]]],
    seed: int = 42,
) -> List[Tuple[str, dict, List[XWRLearnerForestConfig]]]:
    """Randomize cache-group order and job order within each group.

    We still run grouped by cache key so a hidden-state cache is created once,
    but the groups returned by group_configs_by_cache_key are sorted. Shuffle
    them here so cached runs do not execute in dataset/extractor order.
    """
    rng = random.Random(seed)
    randomized_groups = []
    for cache_hash, cache_info, group_configs in cache_groups:
        shuffled_configs = list(group_configs)
        rng.shuffle(shuffled_configs)
        randomized_groups.append((cache_hash, cache_info, shuffled_configs))
    rng.shuffle(randomized_groups)
    return randomized_groups


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
        if config.feature_extractor_type in {
            "frozen_llm_pooler",
            "frozen_llm_token_cnn",
            "frozen_llm_stat_pooler",
            "token_hash_embedding",
            "causal_purity_hash",
            "causal_purity_hash_student",
            "neural_causal_hash_selector",
        }:
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
    contrastive_values = sorted(set(config.contrastive_effect_enabled for config in pending_configs))
    pairwise_values = sorted(set(config.contrastive_pairwise_matching for config in pending_configs))
    document_window_values = sorted(set(getattr(config, "flp_document_window", "tail") for config in pending_configs))
    dr_values = sorted(set(config.effect_dr_weight for config in pending_configs))
    entropy_values = sorted(set(config.effect_attention_entropy_weight for config in pending_configs))
    warm_start_values = sorted(set(config.initialize_effect_from_nuisance for config in pending_configs))
    w_hidden_values = sorted(set(config.wx_nuisance_hidden_dim for config in pending_configs))
    x_hidden_values = sorted(set(config.wx_effect_hidden_dim for config in pending_configs))
    forest_x_modes = sorted(set(config.cf_forest_x_mode for config in pending_configs))
    shared_text_values = sorted(set(config.cf_export_shared_text_features for config in pending_configs))

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
    print(f"Contrastive X stage: {', '.join(str(v) for v in contrastive_values)}")
    print(f"Pairwise propensity matching: {', '.join(str(v) for v in pairwise_values)}")
    print(f"Frozen LLM document windows: {', '.join(document_window_values)}")
    print(f"DR effect weights: {', '.join(str(v) for v in dr_values)}")
    print(f"Effect attention entropy weights: {', '.join(str(v) for v in entropy_values)}")
    print(f"Effect warm-start from nuisance: {', '.join(str(v) for v in warm_start_values)}")
    print(f"W hidden dims: {', '.join(str(v) for v in w_hidden_values)}")
    print(f"X hidden dims: {', '.join(str(v) for v in x_hidden_values)}")
    print(f"Forest X modes: {', '.join(forest_x_modes)}")
    print(f"Shared text features in forest: {', '.join(str(v) for v in shared_text_values)}")
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
            "contrastive_effect_enabled",
            "contrastive_pairwise_matching",
            "contrastive_pair_caliper",
            "contrastive_forest_x_mode",
            "flp_model_name",
            "hlm_model_name",
            "flp_max_length",
            "flp_document_window",
            "use_explicit_features",
            "learning_rate",
            "epochs",
            "cf_n_estimators",
            "cf_min_samples_leaf",
            "effect_dr_weight",
            "effect_attention_entropy_weight",
            "nuisance_potential_weight",
            "initialize_effect_from_nuisance",
            "export_nuisance_potential_features",
            "cf_export_shared_text_features",
            "wx_nuisance_hidden_dim",
            "wx_effect_hidden_dim",
            "cf_forest_x_mode",
            "purity_hash_top_k",
            "purity_hash_alpha",
            "purity_hash_beta",
            "purity_hash_gamma_control",
            "purity_hash_min_count",
            "purity_hash_min_arm_count",
            "purity_hash_candidate_k",
            "purity_hash_student_hidden_dim",
            "purity_hash_student_steps",
            "purity_hash_student_lr",
            "purity_hash_student_threshold",
            "purity_hash_selector_steps",
            "purity_hash_selector_lr",
            "purity_hash_selector_temperature",
            "purity_hash_selector_entropy",
            "purity_hash_selector_threshold",
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
        "--cf-n-estimators",
        type=int,
        default=200,
        help="Number of causal forest trees",
    )
    parser.add_argument(
        "--cf-min-samples-leaf",
        type=int,
        default=5,
        help="Minimum causal forest samples per leaf",
    )
    parser.add_argument(
        "--rlearner-nuisance-folds",
        type=int,
        default=5,
        help="Inner folds for out-of-fold nuisance predictions in each outer train split",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Training and feature-extraction batch size",
    )
    parser.add_argument(
        "--effect-aux-outcome-weight",
        type=float,
        default=0.0,
        help="Weight for treatment-specific factual outcome loss in the effect branch",
    )
    parser.add_argument(
        "--nuisance-potential-weight",
        type=float,
        default=0.0,
        help="Weight for all-neural treatment-specific nuisance outcome heads",
    )
    parser.add_argument(
        "--effect-dr-weight",
        type=float,
        default=0.0,
        help="Weight for doubly robust pseudo-outcome supervision in the effect branch",
    )
    parser.add_argument(
        "--effect-dr-clip",
        type=float,
        default=1.0,
        help="Symmetric clipping bound for DR tau pseudo-outcomes; <=0 disables clipping",
    )
    parser.add_argument(
        "--effect-attention-entropy-weight",
        type=float,
        default=0.0,
        help="Weight for normalized effect-extractor attention entropy minimization",
    )
    parser.add_argument(
        "--initialize-effect-from-nuisance",
        action="store_true",
        help="Warm-start the effect text encoder from the trained nuisance encoder",
    )
    parser.add_argument(
        "--export-nuisance-potential-features",
        action="store_true",
        help="Append all-neural nuisance mu0/mu1/mu1-mu0 summaries to forest X",
    )
    parser.add_argument(
        "--wx-nuisance-hidden-dims",
        type=int,
        nargs="+",
        default=[64],
        help="Grid of W/nuisance branch hidden dimensions",
    )
    parser.add_argument(
        "--wx-effect-hidden-dims",
        type=int,
        nargs="+",
        default=[64],
        help="Grid of X/effect branch hidden dimensions",
    )
    parser.add_argument(
        "--cf-forest-x-modes",
        type=str,
        nargs="+",
        default=["x_hidden_plus_tau"],
        choices=["x_hidden_plus_tau", "x_hidden", "tau"],
        help="Which learned effect representation to pass as causal-forest X",
    )
    parser.add_argument(
        "--cf-export-shared-text-feature-options",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Boolean grid for appending extractor-level shared text features to "
            "forest X/W; default true"
        ),
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
        "--flp-attention-slots",
        type=int,
        nargs="+",
        default=None,
        help="Learned attention slot counts for frozen_llm_pooler",
    )
    parser.add_argument(
        "--flp-document-windows",
        type=str,
        nargs="+",
        choices=["head", "tail", "head_tail"],
        default=None,
        help="Generic token windows for frozen LLM documents",
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
            "Feature extractors to include: frozen_llm_pooler, frozen_llm_token_cnn, "
            "frozen_llm_stat_pooler, token_hash_embedding, causal_purity_hash, "
            "causal_purity_hash_student, neural_causal_hash_selector, "
            "hierarchical_llm, hierarchical_cnn, hierarchical_gru, simple_cnn, "
            "byte_cnn, text_marker. text_marker is diagnostic-only and is not "
            "included unless requested here."
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
        default=[50],
        help="Epoch-count grid",
    )
    parser.add_argument(
        "--explicit-feature-options",
        type=str,
        nargs="+",
        default=None,
        help="Boolean grid for using role-tagged explicit features; default false true",
    )
    parser.add_argument(
        "--contrastive-effect",
        action="store_true",
        help="Use matched contrastive X-stage training instead of per-patient R-loss",
    )
    parser.add_argument("--contrastive-bottleneck-dim", type=int, default=8)
    parser.add_argument("--contrastive-hidden-dim", type=int, default=64)
    parser.add_argument("--contrastive-batch-size", type=int, default=16)
    parser.add_argument("--contrastive-n-propensity-bins", type=int, default=10)
    parser.add_argument("--contrastive-overlap-min", type=float, default=0.05)
    parser.add_argument("--contrastive-overlap-max", type=float, default=0.95)
    parser.add_argument("--contrastive-min-arm-per-bin", type=int, default=2)
    parser.add_argument(
        "--contrastive-pairwise-matching",
        action="store_true",
        help="Use nearest-neighbor treated/control propensity pairs instead of bins",
    )
    parser.add_argument("--contrastive-pair-caliper", type=float, default=0.05)
    parser.add_argument("--contrastive-lambda-factual", type=float, default=1.0)
    parser.add_argument("--contrastive-lambda-contrast", type=float, default=2.0)
    parser.add_argument("--contrastive-lambda-adversary", type=float, default=0.05)
    parser.add_argument("--contrastive-lambda-pair-pull", type=float, default=0.0)
    parser.add_argument("--contrastive-lambda-z-l2", type=float, default=1e-4)
    parser.add_argument("--contrastive-target-clip", type=float, default=1.0)
    parser.add_argument(
        "--contrastive-forest-x-mode",
        type=str,
        default="bottleneck_plus_tau",
        choices=["bottleneck", "tau", "bottleneck_plus_tau"],
    )
    parser.add_argument(
        "--byte-max-chunks",
        type=int,
        nargs="+",
        default=None,
        help="Byte-CNN max chunk counts to include",
    )
    parser.add_argument("--purity-hash-top-k", type=int, default=1)
    parser.add_argument("--purity-hash-alpha", type=float, default=1.0)
    parser.add_argument("--purity-hash-beta", type=float, default=2.0)
    parser.add_argument("--purity-hash-gamma-control", type=float, default=1.0)
    parser.add_argument("--purity-hash-min-count", type=int, default=20)
    parser.add_argument("--purity-hash-min-arm-count", type=int, default=5)
    parser.add_argument("--purity-hash-candidate-k", type=int, default=256)
    parser.add_argument("--purity-hash-student-hidden-dim", type=int, default=64)
    parser.add_argument("--purity-hash-student-steps", type=int, default=700)
    parser.add_argument("--purity-hash-student-lr", type=float, default=0.03)
    parser.add_argument(
        "--purity-hash-student-threshold",
        type=str,
        default="threshold05",
        choices=["none", "threshold05", "train_freq"],
    )
    parser.add_argument("--purity-hash-selector-steps", type=int, default=800)
    parser.add_argument("--purity-hash-selector-lr", type=float, default=1.0)
    parser.add_argument("--purity-hash-selector-temperature", type=float, default=1.0)
    parser.add_argument("--purity-hash-selector-entropy", type=float, default=0.0)
    parser.add_argument(
        "--purity-hash-selector-threshold",
        type=str,
        default="threshold05",
        choices=["none", "threshold05", "argmax"],
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
    shared_text_feature_options = _parse_bool_grid(args.cf_export_shared_text_feature_options)

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
        contrastive_effect_enabled=args.contrastive_effect,
        contrastive_bottleneck_dim=args.contrastive_bottleneck_dim,
        contrastive_hidden_dim=args.contrastive_hidden_dim,
        contrastive_batch_size=args.contrastive_batch_size,
        contrastive_n_propensity_bins=args.contrastive_n_propensity_bins,
        contrastive_overlap_min=args.contrastive_overlap_min,
        contrastive_overlap_max=args.contrastive_overlap_max,
        contrastive_min_arm_per_bin=args.contrastive_min_arm_per_bin,
        contrastive_pairwise_matching=args.contrastive_pairwise_matching,
        contrastive_pair_caliper=args.contrastive_pair_caliper,
        contrastive_lambda_factual=args.contrastive_lambda_factual,
        contrastive_lambda_contrast=args.contrastive_lambda_contrast,
        contrastive_lambda_adversary=args.contrastive_lambda_adversary,
        contrastive_lambda_pair_pull=args.contrastive_lambda_pair_pull,
        contrastive_lambda_z_l2=args.contrastive_lambda_z_l2,
        contrastive_target_clip=args.contrastive_target_clip,
        contrastive_forest_x_mode=args.contrastive_forest_x_mode,
        wx_nuisance_hidden_dims=args.wx_nuisance_hidden_dims,
        wx_effect_hidden_dims=args.wx_effect_hidden_dims,
        cf_forest_x_modes=args.cf_forest_x_modes,
        cf_export_shared_text_feature_options=shared_text_feature_options,
        byte_max_chunks_options=args.byte_max_chunks,
        flp_attention_slot_options=args.flp_attention_slots,
        flp_document_window_options=args.flp_document_windows,
        purity_hash_top_k=args.purity_hash_top_k,
        purity_hash_alpha=args.purity_hash_alpha,
        purity_hash_beta=args.purity_hash_beta,
        purity_hash_gamma_control=args.purity_hash_gamma_control,
        purity_hash_min_count=args.purity_hash_min_count,
        purity_hash_min_arm_count=args.purity_hash_min_arm_count,
        purity_hash_candidate_k=args.purity_hash_candidate_k,
        purity_hash_student_hidden_dim=args.purity_hash_student_hidden_dim,
        purity_hash_student_steps=args.purity_hash_student_steps,
        purity_hash_student_lr=args.purity_hash_student_lr,
        purity_hash_student_threshold=args.purity_hash_student_threshold,
        purity_hash_selector_steps=args.purity_hash_selector_steps,
        purity_hash_selector_lr=args.purity_hash_selector_lr,
        purity_hash_selector_temperature=args.purity_hash_selector_temperature,
        purity_hash_selector_entropy=args.purity_hash_selector_entropy,
        purity_hash_selector_threshold=args.purity_hash_selector_threshold,
    )

    use_cache = args.cache or args.gpu_cache
    configs = []
    for base_config in base_configs:
        for repeat_idx in range(args.n_repeats):
            config = deepcopy(base_config)
            config.repeat_index = repeat_idx
            config.n_folds = args.n_folds
            config.batch_size = args.batch_size
            config.cf_n_estimators = args.cf_n_estimators
            config.cf_min_samples_leaf = args.cf_min_samples_leaf
            config.rlearner_nuisance_folds = args.rlearner_nuisance_folds
            config.effect_aux_outcome_weight = args.effect_aux_outcome_weight
            config.nuisance_potential_weight = args.nuisance_potential_weight
            config.effect_dr_weight = args.effect_dr_weight
            config.effect_dr_clip = args.effect_dr_clip
            config.effect_attention_entropy_weight = args.effect_attention_entropy_weight
            config.initialize_effect_from_nuisance = args.initialize_effect_from_nuisance
            config.export_nuisance_potential_features = args.export_nuisance_potential_features
            if config.effect_dr_weight > 0 and config.nuisance_potential_weight <= 0:
                config.nuisance_potential_weight = 1.0
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
    cache_groups = randomize_execution_groups(
        group_configs_by_cache_key(pending_configs, use_cache)
    )
    logger.info("Randomized execution order across %d cache group(s)", len(cache_groups))

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
