"""Stage 1 text-model training for the researcher all-evidence workflow."""

from __future__ import annotations

import copy
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, TypeVar

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import brier_score_loss, log_loss, mean_squared_error
from sklearn.model_selection import KFold

from ..config import (
    AgenticAttentionVariableForestConfig,
    AgenticFeatureSearchConfig,
    AppliedInferenceConfig,
    BoWViewConfig,
    ExplicitFeatureForestConfig,
    MultiModelForestConfig,
)
from ..models.causal_forest_head import CausalForestHead
from ..utils.calibration import BinaryProbabilityCalibrator
from .htr_modeling_support import (
    _EffectNet,
    _NuisanceNet,
    _binary_log_loss_from_logits,
    _effect_objective_name,
    _logistic_r_logits,
    _logistic_r_tau_from_delta,
    _r_pseudo_outcome,
    _run_crossfit_fold_tasks,
    clip_probability,
)
from .causal_modeling_support import (
    _fit_predict_outcome,
    _fit_predict_propensity,
    _r_loss,
    _safe_roc_auc,
)
from .causal_modeling_support import _hstack_present
from .embedding_contrast_discovery import (
    EmbeddingContrastEvidenceGenerator,
    _binary_labels,
    _binary_mean_difference_direction,
    _canonicalize_svd_component_signs,
    _cluster_local_scientific_config,
    _embedding_cluster_kmeans_parameters,
    _normalize_rows,
    _normalize_rows_configured,
    _normalize_vector,
    _residualize_embeddings,
    _residualize_vector_from_basis,
    _tail_labels,
)
from .discovery_randomness import derive_discovery_seed
from .stage1_agent_context_support import (
    _build_evidence_digest_agent_context,
    _compact_multi_model_agent_context,
)
from .stage1_modeling_support import (
    _agent_visible_metrics,
    _agentic_discovery_handoff_row,
    _align_htr_prediction_frame,
    _clinical_text_examples,
    _htr_effect_metrics,
    _htr_nuisance_metrics,
    _multi_view_importance,
    _normalize_texts,
    _split_is_honest,
    _top_phrase_feature_rows,
    _write_json,
    _write_jsonl,
)
from .htr_evidence_provider import MultiModelHTREvidenceProvider
from .sparse_text_modeling import (
    _binary_split_items,
    _bow_model_params,
    _bow_vectorizer_params,
    _bounded_fold_count,
    _fit_regressor,
    _make_bow_classifier,
    _make_bow_regressor,
    _make_bow_vectorizer,
    _model_feature_scores,
    _top_feature_rows,
)
from .multi_model_pair_uplift import (
    PairUpliftFitResult,
    fit_bow_pair_uplift_train_test,
    fit_htr_pair_uplift_train_test,
)
from .stage1_checkpointing import Stage1CheckpointStore, row_id_identity

logger = logging.getLogger(__name__)

_CheckpointValue = TypeVar("_CheckpointValue")


@dataclass
class _FeatureBundle:
    x_train: np.ndarray
    x_test: np.ndarray
    w_train: np.ndarray
    w_test: np.ndarray
    x_names: List[str]
    w_names: List[str]
    feature_rows: List[Dict[str, Any]]
    prediction_frames: List[pd.DataFrame]
    embedding_rows: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    handoff_evidence: Optional[Dict[str, Any]]
    inner_model_rows: List[Dict[str, Any]]


def _feature_bundle_checkpoint_payload(bundle: _FeatureBundle) -> Dict[str, Any]:
    return {
        "x_train": bundle.x_train,
        "x_test": bundle.x_test,
        "w_train": bundle.w_train,
        "w_test": bundle.w_test,
        "x_names": bundle.x_names,
        "w_names": bundle.w_names,
        "feature_rows": bundle.feature_rows,
        "prediction_frames": bundle.prediction_frames,
        "embedding_rows": bundle.embedding_rows,
        "metrics": bundle.metrics,
        "handoff_evidence": bundle.handoff_evidence,
        "inner_model_rows": bundle.inner_model_rows,
    }


def _feature_bundle_from_checkpoint(payload: Mapping[str, Any]) -> _FeatureBundle:
    return _FeatureBundle(
        x_train=np.asarray(payload["x_train"]),
        x_test=np.asarray(payload["x_test"]),
        w_train=np.asarray(payload["w_train"]),
        w_test=np.asarray(payload["w_test"]),
        x_names=[str(value) for value in payload["x_names"]],
        w_names=[str(value) for value in payload["w_names"]],
        feature_rows=[dict(value) for value in payload["feature_rows"]],
        prediction_frames=list(payload["prediction_frames"]),
        embedding_rows=[dict(value) for value in payload["embedding_rows"]],
        metrics=dict(payload["metrics"]),
        handoff_evidence=(
            None
            if payload.get("handoff_evidence") is None
            else dict(payload["handoff_evidence"])
        ),
        inner_model_rows=[dict(value) for value in payload["inner_model_rows"]],
    )


def _pair_result_checkpoint_payload(result: PairUpliftFitResult) -> Dict[str, Any]:
    return {
        "train_delta_logit": result.train_delta_logit,
        "test_delta_logit": result.test_delta_logit,
        "train_pred_prob": result.train_pred_prob,
        "test_pred_prob": result.test_pred_prob,
        "train_n_controls": result.train_n_controls,
        "test_n_controls": result.test_n_controls,
        "feature_importance": result.feature_importance,
        "evidence_rows": result.evidence_rows,
        "attention_rows": result.attention_rows,
        "prediction_frame": result.prediction_frame,
        "metrics": result.metrics,
    }


def _pair_result_from_checkpoint(payload: Mapping[str, Any]) -> PairUpliftFitResult:
    return PairUpliftFitResult(
        train_delta_logit=np.asarray(payload["train_delta_logit"], dtype=float),
        test_delta_logit=np.asarray(payload["test_delta_logit"], dtype=float),
        train_pred_prob=np.asarray(payload["train_pred_prob"], dtype=float),
        test_pred_prob=np.asarray(payload["test_pred_prob"], dtype=float),
        train_n_controls=np.asarray(payload["train_n_controls"], dtype=float),
        test_n_controls=np.asarray(payload["test_n_controls"], dtype=float),
        feature_importance=dict(payload["feature_importance"]),
        evidence_rows=[dict(value) for value in payload["evidence_rows"]],
        attention_rows=[dict(value) for value in payload["attention_rows"]],
        prediction_frame=payload["prediction_frame"],
        metrics=dict(payload["metrics"]),
    )


def _validate_train_test_prediction_tuple(
    result: Any,
    train_rows: int,
    test_rows: int,
    label: str,
) -> None:
    if not isinstance(result, (list, tuple)) or len(result) != 3:
        raise ValueError(f"{label} checkpoint must contain train, test, and evidence rows")
    if len(np.asarray(result[0])) != int(train_rows):
        raise ValueError(f"{label} checkpoint fit row count changed")
    if len(np.asarray(result[1])) != int(test_rows):
        raise ValueError(f"{label} checkpoint held-out row count changed")
    if not isinstance(result[2], list):
        raise ValueError(f"{label} checkpoint evidence must be a list")


def _validate_pair_result_payload(
    payload: Mapping[str, Any],
    train_rows: int,
    test_rows: int,
    label: str,
) -> None:
    for name in ("train_delta_logit", "train_pred_prob", "train_n_controls"):
        if len(np.asarray(payload[name])) != int(train_rows):
            raise ValueError(f"{label} checkpoint {name} fit row count changed")
    for name in ("test_delta_logit", "test_pred_prob", "test_n_controls"):
        if len(np.asarray(payload[name])) != int(test_rows):
            raise ValueError(f"{label} checkpoint {name} held-out row count changed")
    if not isinstance(payload.get("prediction_frame"), pd.DataFrame):
        raise ValueError(f"{label} checkpoint prediction_frame is not a DataFrame")


def _validate_nuisance_ensemble_payload(
    payload: Mapping[str, Any],
    train_rows: int,
    test_rows: int,
) -> None:
    for name in (
        "e_train",
        "m_train",
        "e_train_clip",
        "t_resid",
        "y_resid",
        "pseudo_target",
        "r_weight",
    ):
        if len(np.asarray(payload[name])) != int(train_rows):
            raise ValueError(f"nuisance ensemble checkpoint {name} fit row count changed")
    for name in ("e_test", "m_test"):
        if len(np.asarray(payload[name])) != int(test_rows):
            raise ValueError(f"nuisance ensemble checkpoint {name} held-out row count changed")
    train_predictions = payload.get("train_predictions")
    test_predictions = payload.get("test_predictions")
    if not isinstance(train_predictions, pd.DataFrame) or len(train_predictions) != train_rows:
        raise ValueError("nuisance ensemble checkpoint train prediction rows changed")
    if not isinstance(test_predictions, pd.DataFrame) or len(test_predictions) != test_rows:
        raise ValueError("nuisance ensemble checkpoint held-out prediction rows changed")


def _validate_mapping_checkpoint(value: Any, label: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} checkpoint is not a mapping")


def _validate_embedding_checkpoint_payload(
    payload: Mapping[str, Any],
    train_rows: int,
    test_rows: int,
) -> None:
    for family in ("w_features", "x_features"):
        features = payload.get(family)
        if not isinstance(features, list):
            raise ValueError(f"embedding checkpoint {family} is not a list")
        for feature in features:
            if len(np.asarray(feature["train"])) != int(train_rows):
                raise ValueError("embedding checkpoint fit feature row count changed")
            if len(np.asarray(feature["test"])) != int(test_rows):
                raise ValueError("embedding checkpoint held-out feature row count changed")
    if not isinstance(payload.get("metadata"), list):
        raise ValueError("embedding checkpoint metadata is not a list")


def _validate_prediction_fold_payload(
    payload: Mapping[str, Any],
    heldout_rows: int,
    test_rows: int,
    label: str,
) -> None:
    if len(np.asarray(payload["heldout_pos"])) != int(heldout_rows):
        raise ValueError(f"{label} checkpoint held-out positions changed")
    if len(np.asarray(payload["heldout_pred"])) != int(heldout_rows):
        raise ValueError(f"{label} checkpoint held-out predictions changed")
    if len(np.asarray(payload["test_pred"])) != int(test_rows):
        raise ValueError(f"{label} checkpoint outer-test predictions changed")


@dataclass(frozen=True)
class MultiModelForestStage1ParallelPlan:
    """Resolved worker budget for exact Stage 1 discovery contexts."""

    cpus_total: int
    gpu_ids: List[int]
    htr_jobs_per_gpu: int
    htr_enabled: bool
    embedding_enabled: bool
    htr_slots: int
    reserved_htr_cpus: int
    cpu_loky_workers: int
    context_workers: int
    htr_inner_jobs_per_outer: int
    htr_device_slots: List[Optional[int]]

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "cpus_total": self.cpus_total,
            "cpu_loky_workers": self.cpu_loky_workers,
            "gpu_ids": self.gpu_ids,
            "htr_jobs_per_gpu": self.htr_jobs_per_gpu,
            "htr_slots": self.htr_slots,
            "htr_inner_jobs_per_outer": self.htr_inner_jobs_per_outer,
            "context_workers": self.context_workers,
            "embedding_enabled": self.embedding_enabled,
            "htr_enabled": self.htr_enabled,
        }


def resolve_multi_model_forest_stage1_parallel_plan(
    *,
    cpus_total: Optional[int],
    num_workers: int,
    gpu_ids: Optional[Sequence[int]],
    htr_jobs_per_gpu: int,
    htr_enabled: bool,
    embedding_enabled: bool,
) -> MultiModelForestStage1ParallelPlan:
    """Resolve the public CPU/GPU budget for exact Stage 1 context fits."""

    total = int(cpus_total if cpus_total is not None else num_workers)
    total = max(1, total)
    gpus = [int(gpu_id) for gpu_id in (gpu_ids or [])]
    jobs_per_gpu = max(1, int(htr_jobs_per_gpu))
    htr_slots = len(gpus) * jobs_per_gpu if htr_enabled and gpus else (1 if htr_enabled else 0)
    reserved = min(total - 1, htr_slots) if htr_slots > 0 else 0
    cpu_workers = max(1, total - reserved)
    if htr_enabled and gpus:
        context_workers = max(1, min(total, len(gpus)))
        htr_inner_jobs = jobs_per_gpu
    elif htr_enabled:
        context_workers = 1
        htr_inner_jobs = 1
    else:
        context_workers = cpu_workers
        htr_inner_jobs = 1
    device_slots: List[Optional[int]] = []
    if htr_enabled and gpus:
        for _job_index in range(jobs_per_gpu):
            for gpu_id in gpus:
                device_slots.append(int(gpu_id))
    if not device_slots:
        device_slots = [None] * context_workers
    return MultiModelForestStage1ParallelPlan(
        cpus_total=total,
        gpu_ids=gpus,
        htr_jobs_per_gpu=jobs_per_gpu,
        htr_enabled=bool(htr_enabled),
        embedding_enabled=bool(embedding_enabled),
        htr_slots=int(htr_slots),
        reserved_htr_cpus=int(reserved),
        cpu_loky_workers=int(cpu_workers),
        context_workers=int(max(1, context_workers)),
        htr_inner_jobs_per_outer=int(max(1, htr_inner_jobs)),
        htr_device_slots=device_slots,
    )


def config_for_multi_model_forest_handoff(
    config: AppliedInferenceConfig,
) -> AppliedInferenceConfig:
    """Return an isolated configuration for one exact handoff context fit."""

    cfg = copy.deepcopy(config)
    mm_config = getattr(cfg.architecture, "multi_model_forest", None)
    if mm_config is None:
        mm_config = MultiModelForestConfig()
        cfg.architecture.multi_model_forest = mm_config
    cfg.architecture.model_type = "multi_model_forest"
    mm_config = copy.deepcopy(mm_config)
    mm_config.outer_parallelism = "1"
    mm_config.candidate_consistency_parallelism = "1"
    mm_config.fold_parallelism = "auto"
    mm_config.bow_fold_parallelism = "auto"
    # Whole contexts already occupy separate loky processes. Threads let the
    # independent BoW folds share each lane's data without another process tree.
    mm_config.bow_parallel_backend = "threads"
    cfg.architecture.multi_model_forest = copy.deepcopy(mm_config)
    cfg.architecture.multi_model_agentic_forest = mm_config
    avf_config = getattr(cfg.architecture, "agentic_attention_variable_forest", None)
    if avf_config is None:
        avf_config = AgenticAttentionVariableForestConfig()
        cfg.architecture.agentic_attention_variable_forest = avf_config
    avf_config.fold_parallelism = "1"
    return cfg


def run_multi_model_forest_handoff_contexts(
    *,
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    contexts: Sequence[Dict[str, Any]],
    handoff_dir: Path,
    plan: MultiModelForestStage1ParallelPlan,
    base_device: torch.device,
    checkpoint_store: Optional[Stage1CheckpointStore] = None,
) -> List[Dict[str, Any]]:
    """Fit all enabled Stage 1 architectures in each exact discovery context."""

    if not contexts:
        return []
    if checkpoint_store is not None and len(contexts) != 1:
        raise ValueError("one Stage 1 checkpoint store may only serve one exact context")
    n_workers = max(1, min(int(plan.context_workers), len(contexts)))
    slots = _handoff_worker_slots(plan, n_workers, base_device)
    cpu_workers = _handoff_cpu_worker_budgets(plan.cpus_total, n_workers)
    shards = [[] for _ in range(n_workers)]
    for index, context in enumerate(contexts):
        shards[index % n_workers].append(context)
    logger.info(
        "Precomputing multi-model forest Stage 1 contexts=%s loky_workers=%s "
        "cpu_workers_per_lane=%s slots=%s",
        len(contexts),
        n_workers,
        cpu_workers,
        slots,
    )
    if n_workers <= 1:
        return _run_handoff_context_shard(
            dataset=dataset,
            config=config,
            shard=shards[0],
            handoff_dir=handoff_dir,
            shard_index=0,
            device=str(slots[0][0]),
            gpu_ids=slots[0][1],
            num_workers=cpu_workers[0],
            checkpoint_store=checkpoint_store,
        )
    shard_rows = Parallel(
        n_jobs=n_workers,
        backend="loky",
        batch_size=1,
        pre_dispatch="all",
    )(
        delayed(_run_handoff_context_shard)(
            dataset=dataset,
            config=config,
            shard=shard,
            handoff_dir=handoff_dir,
            shard_index=shard_index,
            device=str(slots[shard_index][0]),
            gpu_ids=slots[shard_index][1],
            num_workers=cpu_workers[shard_index],
            checkpoint_store=None,
        )
        for shard_index, shard in enumerate(shards)
        if shard
    )
    return [row for rows in shard_rows for row in rows]


def _handoff_cpu_worker_budgets(cpus_total: int, lane_count: int) -> List[int]:
    """Divide an overall CPU budget among concurrent handoff lanes."""

    lane_count = int(lane_count)
    if lane_count < 1:
        raise ValueError("lane_count must be positive")
    total = max(lane_count, int(cpus_total), 1)
    per_lane, remainder = divmod(total, lane_count)
    return [per_lane + (1 if lane_index < remainder else 0) for lane_index in range(lane_count)]


def _handoff_worker_slots(
    plan: MultiModelForestStage1ParallelPlan,
    n_workers: int,
    base_device: torch.device,
) -> List[Tuple[torch.device, Optional[List[int]]]]:
    slots: List[Tuple[torch.device, Optional[List[int]]]] = []
    if plan.htr_enabled and plan.gpu_ids:
        gpu_slots = plan.htr_device_slots or plan.gpu_ids
        for index in range(n_workers):
            gpu_id = int(gpu_slots[index % len(gpu_slots)])
            slots.append((torch.device(f"cuda:{gpu_id}"), [gpu_id]))
        return slots
    for _ in range(n_workers):
        slots.append((base_device, None))
    return slots


def _run_handoff_context_shard(
    *,
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    shard: Sequence[Dict[str, Any]],
    handoff_dir: Path,
    shard_index: int,
    device: str,
    gpu_ids: Optional[List[int]],
    num_workers: int,
    checkpoint_store: Optional[Stage1CheckpointStore] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with _serial_torch_worker_environment():
        for context in shard:
            train_idx = np.asarray(context["train_idx"], dtype=int)
            runner_kwargs: Dict[str, Any] = {
                "dataset": dataset,
                "config": config_for_multi_model_forest_handoff(config),
                "output_path": (
                    Path(handoff_dir)
                    / "worker_artifacts"
                    / f"shard_{int(shard_index):03d}"
                    / f"fold_{int(context['fold_key']):06d}"
                    / "predictions.parquet"
                ),
                "device": torch.device(device),
                "gpu_ids": gpu_ids,
                "num_workers": max(1, int(num_workers)),
            }
            if checkpoint_store is not None:
                runner_kwargs["checkpoint_store"] = checkpoint_store
            runner = MultiModelForestStage1Runner(**runner_kwargs)
            heldout_idx = np.asarray(context["heldout_idx"], dtype=int)
            logger.info(
                "Precomputing handoff context fold_key=%s scope=%s rows=%s device=%s",
                context["fold_key"],
                context["scope"],
                len(train_idx),
                device,
            )
            rows.append(
                runner.build_discovery_handoff_row(
                    train_idx=train_idx,
                    heldout_idx=heldout_idx,
                    fold_key=int(context["fold_key"]),
                    outer_fold=int(context["outer_fold"]),
                    scope=str(context["scope"]),
                    inner_fold=context.get("inner_fold"),
                    heldout_rows=context.get("heldout_rows"),
                )
            )
    return rows


@contextmanager
def _serial_torch_worker_environment():
    """Constrain nested numerical pools inside one Stage 1 context worker."""

    from threadpoolctl import threadpool_limits

    env_keys = [
        "OCI_AVF_DATALOADER_WORKERS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]
    previous = {key: os.environ.get(key) for key in env_keys}
    os.environ["OCI_AVF_DATALOADER_WORKERS"] = "0"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    old_threads = None
    try:
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
    except Exception:
        old_threads = None
    try:
        with threadpool_limits(limits=1):
            yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if old_threads is not None:
            try:
                torch.set_num_threads(old_threads)
            except Exception:
                pass


def _run_stage1_outer_fold_job(
    *,
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    output_path: Path,
    outer_fold: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    device: str,
    gpu_ids: Optional[Sequence[int]],
    num_workers: int,
    htr_dataloader_workers: Optional[int] = None,
) -> Dict[str, Any]:
    previous_dataloader_workers = os.environ.get("OCI_AVF_DATALOADER_WORKERS")
    if htr_dataloader_workers is not None:
        os.environ["OCI_AVF_DATALOADER_WORKERS"] = str(max(0, int(htr_dataloader_workers)))
    fold_runner = MultiModelForestStage1Runner(
        dataset=dataset.drop(columns=["_oci_row_id"], errors="ignore"),
        config=copy.deepcopy(config),
        output_path=output_path,
        device=torch.device(device),
        gpu_ids=gpu_ids,
        num_workers=num_workers,
    )
    try:
        predictions = fold_runner._run_one_analysis_split(
            outer_fold=outer_fold,
            train_idx=train_idx,
            test_idx=test_idx,
        )
        return {
            "outer_fold": int(outer_fold),
            "predictions": predictions,
            "feature_manifest_rows": fold_runner.feature_manifest_rows,
            "source_prediction_frames": fold_runner.source_prediction_frames,
            "embedding_feature_rows": fold_runner.embedding_feature_rows,
            "outer_metric_rows": fold_runner.outer_metric_rows,
            "agentic_handoff_rows": fold_runner.agentic_handoff_rows,
            "inner_model_evidence_rows": fold_runner.inner_model_evidence_rows,
        }
    finally:
        if htr_dataloader_workers is not None:
            if previous_dataloader_workers is None:
                os.environ.pop("OCI_AVF_DATALOADER_WORKERS", None)
            else:
                os.environ["OCI_AVF_DATALOADER_WORKERS"] = previous_dataloader_workers


def run_multi_model_forest_stage1(
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    output_path: Path,
    device=None,
    gpu_ids: Optional[Sequence[int]] = None,
    num_workers: int = 1,
    embedding_provider: Optional[Any] = None,
    htr_evidence_provider: Optional[Any] = None,
) -> None:
    """Run the non-agentic multi-model W/X causal forest path."""
    runner = MultiModelForestStage1Runner(
        dataset=dataset,
        config=config,
        output_path=output_path,
        device=device,
        gpu_ids=gpu_ids,
        num_workers=num_workers,
        embedding_provider=embedding_provider,
        htr_evidence_provider=htr_evidence_provider,
    )
    runner.run()


class MultiModelForestStage1HTRProvider(MultiModelHTREvidenceProvider):
    """HTR adapter with full outer-train -> outer-test prediction helpers."""

    def _active_checkpoint_store(self) -> Optional[Stage1CheckpointStore]:
        if getattr(self, "native_capture_sink", None) is not None:
            return None
        if getattr(self, "native_pair_capture_sink", None) is not None:
            return None
        return getattr(self, "checkpoint_store", None)

    @contextmanager
    def _temporary_effect_objective(self, objective: str):
        runner = self._runner
        if runner is None:
            raise RuntimeError("HTR runner has not been initialized")
        previous = getattr(runner.avf_config, "effect_objective", "pseudo_outcome_mse")
        runner.avf_config.effect_objective = str(objective)
        try:
            yield
        finally:
            runner.avf_config.effect_objective = previous

    def fit_effect_variant(
        self,
        discovery_df: pd.DataFrame,
        nuisance_predictions: pd.DataFrame,
        outer_fold: int,
        *,
        effect_objective: str,
    ) -> Dict[str, Any]:
        runner = self._ensure_runner(discovery_df)
        with self._temporary_effect_objective(effect_objective):
            return runner._crossfit_effect(discovery_df, nuisance_predictions, outer_fold)

    def fit_pair_uplift_inner_ensemble_predict(
        self,
        *,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        texts_train: Sequence[str],
        texts_test: Sequence[str],
        y_train: np.ndarray,
        t_train: np.ndarray,
        e_train: np.ndarray,
        m_train: np.ndarray,
        e_test: np.ndarray,
        m_test: np.ndarray,
        outer_fold: int,
        propensity_caliper: float,
        outcome_caliper: float,
        max_controls_per_candidate: int,
        nearest_fallback_controls: int,
        max_attention_pairs: int,
    ) -> Any:
        runner = self._ensure_runner(train_df)
        return fit_htr_pair_uplift_train_test(
            runner=runner,
            train_df=train_df,
            test_df=test_df,
            texts_train=texts_train,
            texts_test=texts_test,
            y_train=y_train,
            t_train=t_train,
            e_train=e_train,
            m_train=m_train,
            e_test=e_test,
            m_test=m_test,
            outer_fold=outer_fold,
            effect_folds=self.config.architecture.multi_model_forest.effect_folds,
            propensity_caliper=propensity_caliper,
            outcome_caliper=outcome_caliper,
            max_controls_per_candidate=max_controls_per_candidate,
            nearest_fallback_controls=nearest_fallback_controls,
            max_attention_pairs=max_attention_pairs,
            native_capture_sink=getattr(self, "native_pair_capture_sink", None),
            checkpoint_store=self._active_checkpoint_store(),
            checkpoint_dependencies=("assembly/nuisance_ensemble",),
        )

    def fit_nuisance_inner_ensemble_predict(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        outer_fold: int,
    ) -> Dict[str, Any]:
        runner = self._ensure_runner(train_df)
        folds = _bounded_fold_count(runner.avf_config.nuisance_folds, len(train_df))
        predictions = pd.DataFrame(
            {
                "_oci_row_id": train_df["_oci_row_id"].to_numpy(),
                "outer_fold": int(outer_fold),
                "e_hat": np.nan,
                "e_hat_raw": np.nan,
                "m_hat": np.nan,
                "m_hat_raw": np.nan,
                "y_residual": np.nan,
                "t_residual": np.nan,
                "r_pseudo_outcome": np.nan,
                "r_loss_at_zero_tau": np.nan,
                "nuisance_fold": np.nan,
            }
        )
        split_items = list(
            enumerate(
                KFold(n_splits=folds, shuffle=True, random_state=10_000 + outer_fold).split(
                    train_df
                ),
                start=1,
            )
        )

        def compute_fold(fold: int, fit_pos: np.ndarray, heldout_pos: np.ndarray):
            model = None
            fit_pos = np.asarray(fit_pos, dtype=int)
            heldout_pos = np.asarray(heldout_pos, dtype=int)
            try:
                model = _NuisanceNet(
                    extractor=runner._create_extractor(),
                    hidden_dim=getattr(
                        runner.config.architecture,
                        "htr_prediction_head_hidden_dim",
                        64,
                    ),
                    outcome_type=runner.config.outcome_type,
                ).to(runner.device)
                runner._train_nuisance_model(
                    model,
                    train_df,
                    fit_pos,
                    outer_fold=outer_fold,
                    fold=fold,
                    total_folds=folds,
                )
                fit_df = train_df.iloc[fit_pos]
                heldout = train_df.iloc[heldout_pos]
                e_fit_raw, m_fit_raw = runner._predict_nuisance_model(model, fit_df)
                e_raw, m_raw = runner._predict_nuisance_model(model, heldout)
                e_test_raw, m_test_raw = runner._predict_nuisance_model(model, test_df)
                prop_calibrator = BinaryProbabilityCalibrator.fit(
                    e_fit_raw,
                    fit_df[runner.config.treatment_column].to_numpy(dtype=float),
                    method=runner.avf_config.nuisance_calibration,
                )
                e_hat = prop_calibrator.transform(e_raw)
                e_test = prop_calibrator.transform(e_test_raw)
                outcome_calibrator = None
                if runner.config.outcome_type == "continuous":
                    m_hat = m_raw
                    m_test = m_test_raw
                else:
                    outcome_calibrator = BinaryProbabilityCalibrator.fit(
                        m_fit_raw,
                        fit_df[runner.config.outcome_column].to_numpy(dtype=float),
                        method=runner.avf_config.nuisance_calibration,
                    )
                    m_hat = outcome_calibrator.transform(m_raw)
                    m_test = outcome_calibrator.transform(m_test_raw)
                y = heldout[runner.config.outcome_column].to_numpy(dtype=float)
                t = heldout[runner.config.treatment_column].to_numpy(dtype=float)
                y_resid = y - m_hat
                t_resid = t - e_hat
                native_capture = getattr(self, "native_capture_sink", None)
                if native_capture is not None:
                    native_capture.record_nuisance_fold(
                        model=model,
                        train_df=train_df,
                        test_df=test_df,
                        fit_pos=fit_pos,
                        validation_pos=heldout_pos,
                        fold=fold,
                        fit_e_raw=e_fit_raw,
                        fit_m_raw=m_fit_raw,
                        validation_e_raw=e_raw,
                        validation_m_raw=m_raw,
                        validation_e_hat=e_hat,
                        validation_m_hat=m_hat,
                        heldout_e_raw=e_test_raw,
                        heldout_m_raw=m_test_raw,
                        heldout_e_hat=e_test,
                        heldout_m_hat=m_test,
                        propensity_calibrator=prop_calibrator,
                        outcome_calibrator=outcome_calibrator,
                    )
                fold_attention = runner._attention_evidence(
                    model.extractor,
                    heldout,
                    fold=fold,
                    outer_fold=outer_fold,
                    stage="nuisance",
                    extra={
                        "e_hat": e_hat,
                        "e_hat_raw": e_raw,
                        "m_hat": m_hat,
                        "m_hat_raw": m_raw,
                        "y_residual": y_resid,
                        "t_residual": t_resid,
                    },
                )
                return {
                    "fold": int(fold),
                    "heldout_pos": heldout_pos,
                    "e_hat": e_hat,
                    "e_hat_raw": e_raw,
                    "m_hat": m_hat,
                    "m_hat_raw": m_raw,
                    "y_resid": y_resid,
                    "t_resid": t_resid,
                    "test_e_hat": e_test,
                    "test_m_hat": m_test,
                    "attention": fold_attention,
                    "evidence": {
                        "outer_fold": int(outer_fold),
                        "inner_fold": int(fold),
                        "source_family": "htr",
                        "objective": "nuisance",
                        "target_name": "treatment_outcome_nuisance",
                        "train_rows": int(len(fit_pos)),
                        "heldout_rows": int(len(heldout_pos)),
                        "outer_test_rows": int(len(test_df)),
                        "heldout_treatment_auroc": _safe_roc_auc(t, e_hat),
                        "heldout_outcome_auroc": (
                            _safe_roc_auc(y, m_hat)
                            if runner.config.outcome_type != "continuous"
                            else None
                        ),
                        "prediction_provenance": "inner_fold_model_heldout_and_outer_test",
                    },
                }
            finally:
                if model is not None:
                    runner._cleanup_model(model)

        def run_fold(fold: int, fit_pos: np.ndarray, heldout_pos: np.ndarray):
            store = self._active_checkpoint_store()
            if store is None:
                return compute_fold(fold, fit_pos, heldout_pos)
            fit_pos = np.asarray(fit_pos, dtype=int)
            heldout_pos = np.asarray(heldout_pos, dtype=int)

            def validate(result: Mapping[str, Any]) -> None:
                if len(np.asarray(result["heldout_pos"])) != len(heldout_pos):
                    raise ValueError("HTR nuisance checkpoint held-out row count changed")
                for name in ("e_hat", "e_hat_raw", "m_hat", "m_hat_raw"):
                    if len(np.asarray(result[name])) != len(heldout_pos):
                        raise ValueError(f"HTR nuisance checkpoint {name} row count changed")
                for name in ("test_e_hat", "test_m_hat"):
                    if len(np.asarray(result[name])) != len(test_df):
                        raise ValueError(f"HTR nuisance checkpoint {name} row count changed")

            return store.run(
                f"htr/nuisance/fold_{int(fold):03d}",
                lambda: compute_fold(fold, fit_pos, heldout_pos),
                parameters={
                    "outer_fold": int(outer_fold),
                    "inner_fold": int(fold),
                    "total_folds": int(folds),
                    "objective": "treatment_outcome_nuisance",
                },
                rows={
                    "fit": row_id_identity(
                        train_df.iloc[fit_pos]["_oci_row_id"].to_numpy()
                    ),
                    "validation": row_id_identity(
                        train_df.iloc[heldout_pos]["_oci_row_id"].to_numpy()
                    ),
                    "heldout": row_id_identity(test_df["_oci_row_id"].to_numpy()),
                },
                validator=validate,
                deterministic_seed=True,
            ).value

        store = self._active_checkpoint_store()
        # Torch RNG is process-global.  Serial checkpointed leaves make the
        # per-leaf seed deterministic without retaining epoch/model state.
        n_jobs = 1 if store is not None else runner._fold_n_jobs(folds)
        fold_results = _run_crossfit_fold_tasks(
            run_fold,
            split_items,
            n_jobs,
            device_context=runner._device_context_for_inner_fold,
        )
        attention_rows: List[Dict[str, Any]] = []
        test_e = []
        test_m = []
        inner_model_rows = []
        for result in fold_results:
            heldout_pos = result["heldout_pos"]
            predictions.loc[heldout_pos, "e_hat"] = result["e_hat"]
            predictions.loc[heldout_pos, "e_hat_raw"] = result["e_hat_raw"]
            predictions.loc[heldout_pos, "m_hat"] = result["m_hat"]
            predictions.loc[heldout_pos, "m_hat_raw"] = result["m_hat_raw"]
            predictions.loc[heldout_pos, "y_residual"] = result["y_resid"]
            predictions.loc[heldout_pos, "t_residual"] = result["t_resid"]
            predictions.loc[heldout_pos, "r_pseudo_outcome"] = _r_pseudo_outcome(
                result["y_resid"],
                result["t_resid"],
            )
            predictions.loc[heldout_pos, "r_loss_at_zero_tau"] = result["y_resid"] ** 2
            predictions.loc[heldout_pos, "nuisance_fold"] = result["fold"]
            attention_rows.extend(result["attention"])
            test_e.append(np.asarray(result["test_e_hat"], dtype=float))
            test_m.append(np.asarray(result["test_m_hat"], dtype=float))
            inner_model_rows.append(result["evidence"])
        test_predictions = pd.DataFrame(
            {
                "_oci_row_id": test_df["_oci_row_id"].to_numpy(),
                "outer_fold": int(outer_fold),
                "e_hat": np.nanmean(np.vstack(test_e), axis=0),
                "m_hat": np.nanmean(np.vstack(test_m), axis=0),
                "model_family": "htr",
                "view_name": "htr_nuisance",
                "target_source": "htr_nuisance_inner_ensemble",
            }
        )
        return {
            "train": {"predictions": predictions, "attention": attention_rows},
            "test_predictions": test_predictions,
            "inner_model_rows": inner_model_rows,
        }

    def fit_effect_variant_inner_ensemble_predict(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        nuisance_predictions: pd.DataFrame,
        outer_fold: int,
        *,
        effect_objective: str,
        test_nuisance_predictions: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        runner = self._ensure_runner(train_df)
        folds = _bounded_fold_count(runner.avf_config.effect_folds, len(train_df))
        with self._temporary_effect_objective(effect_objective):
            effect_objective = _effect_objective_name(runner.avf_config)
            r_df = train_df[["_oci_row_id"]].merge(
                nuisance_predictions.copy(),
                on="_oci_row_id",
                how="left",
                sort=False,
            )
            e = r_df["e_hat"].to_numpy(dtype=float)
            m = r_df["m_hat"].to_numpy(dtype=float)
            y = train_df[runner.config.outcome_column].to_numpy(dtype=float)
            t = train_df[runner.config.treatment_column].to_numpy(dtype=float)
            e_clipped = np.clip(e, runner.avf_config.e_clip, 1.0 - runner.avf_config.e_clip)
            m_clipped = clip_probability(m)
            t_resid = t - e_clipped
            y_resid = y - m
            r_pseudo_outcome = _r_pseudo_outcome(y_resid, t_resid)
            r_stage_min = float(getattr(runner.avf_config, "r_stage_min_propensity", 0.0))
            r_stage_max = float(getattr(runner.avf_config, "r_stage_max_propensity", 1.0))
            train_eligible = np.isfinite(e) & (e >= r_stage_min) & (e <= r_stage_max)
            if effect_objective == "pseudo_outcome_mse":
                train_eligible = train_eligible & np.isfinite(r_pseudo_outcome)
            r_df["tau_hat_r_stage"] = np.nan
            r_df["tau_logit_modifier"] = np.nan
            r_df["r_loss"] = np.nan
            r_df["effect_loss"] = np.nan
            r_df["effect_loss_at_zero_tau"] = (
                y_resid**2
                if effect_objective != "logistic_r_loss"
                else _binary_log_loss_from_logits(_logistic_r_logits(0, t, e_clipped, m_clipped), y)
            )
            r_df["effect_fold"] = np.nan
            r_df["r_stage_train_eligible"] = train_eligible
            r_df["effect_objective"] = effect_objective
            r_df["r_pseudo_outcome"] = r_pseudo_outcome

            test_e = None
            test_m = None
            if test_nuisance_predictions is not None:
                test_e = np.asarray(test_nuisance_predictions["e_hat"], dtype=float)
                test_m = np.asarray(test_nuisance_predictions["m_hat"], dtype=float)
            split_items = list(
                enumerate(
                    KFold(n_splits=folds, shuffle=True, random_state=20_000 + outer_fold).split(
                        train_df
                    ),
                    start=1,
                )
            )

            def compute_fold(fold: int, fit_pos: np.ndarray, heldout_pos: np.ndarray):
                model = None
                fit_pos = np.asarray(fit_pos, dtype=int)
                heldout_pos = np.asarray(heldout_pos, dtype=int)
                eligible_fit_pos = fit_pos[train_eligible[fit_pos]]
                if len(eligible_fit_pos) < 1:
                    raise ValueError(
                        "No rows remain for HTR R-stage inner fold "
                        f"{fold} after applying propensity bounds"
                    )
                try:
                    model = _EffectNet(
                        extractor=runner._create_extractor(),
                        hidden_dim=getattr(
                            runner.config.architecture,
                            "htr_prediction_head_hidden_dim",
                            64,
                        ),
                    ).to(runner.device)
                    runner._train_effect_model(
                        model,
                        train_df,
                        eligible_fit_pos,
                        y,
                        t,
                        e_clipped,
                        m_clipped,
                        y_resid,
                        t_resid,
                        outer_fold=outer_fold,
                        fold=fold,
                        total_folds=folds,
                    )
                    heldout = train_df.iloc[heldout_pos]
                    raw_effect = runner._predict_effect_model(model, heldout)
                    raw_test = runner._predict_effect_model(model, test_df)
                    if effect_objective == "logistic_r_loss":
                        tau_logit_modifier = raw_effect
                        tau_hat = _logistic_r_tau_from_delta(
                            tau_logit_modifier,
                            e_clipped[heldout_pos],
                            m_clipped[heldout_pos],
                            e_clip=runner.avf_config.e_clip,
                        )
                        if test_e is None or test_m is None:
                            test_tau = raw_test
                        else:
                            test_tau = _logistic_r_tau_from_delta(
                                raw_test,
                                np.clip(
                                    test_e, runner.avf_config.e_clip, 1.0 - runner.avf_config.e_clip
                                ),
                                clip_probability(test_m),
                                e_clip=runner.avf_config.e_clip,
                            )
                        heldout_effect_loss = _binary_log_loss_from_logits(
                            _logistic_r_logits(
                                tau_logit_modifier,
                                t[heldout_pos],
                                e_clipped[heldout_pos],
                                m_clipped[heldout_pos],
                                e_clip=runner.avf_config.e_clip,
                            ),
                            y[heldout_pos],
                        )
                    else:
                        tau_hat = raw_effect
                        test_tau = raw_test
                        tau_logit_modifier = np.full(len(heldout_pos), np.nan)
                        heldout_effect_loss = (
                            (tau_hat - r_pseudo_outcome[heldout_pos]) ** 2
                            if effect_objective == "pseudo_outcome_mse"
                            else (y_resid[heldout_pos] - tau_hat * t_resid[heldout_pos]) ** 2
                        )
                    heldout_r_loss = (y_resid[heldout_pos] - tau_hat * t_resid[heldout_pos]) ** 2
                    native_capture = getattr(self, "native_capture_sink", None)
                    if native_capture is not None:
                        native_capture.record_effect_fold(
                            model=model,
                            train_df=train_df,
                            test_df=test_df,
                            fit_pos=fit_pos,
                            eligible_fit_pos=eligible_fit_pos,
                            validation_pos=heldout_pos,
                            fold=fold,
                            effect_objective=effect_objective,
                            treatment=t,
                            outcome=y,
                            e_hat=e,
                            m_hat=m,
                            validation_raw_effect=raw_effect,
                            validation_tau=tau_hat,
                            validation_r_loss=heldout_r_loss,
                            validation_effect_loss=heldout_effect_loss,
                            heldout_raw_effect=raw_test,
                            heldout_tau=test_tau,
                            r_stage_min_propensity=r_stage_min,
                            r_stage_max_propensity=r_stage_max,
                        )
                    fold_attention = runner._attention_evidence(
                        model.extractor,
                        heldout,
                        fold=fold,
                        outer_fold=outer_fold,
                        stage="effect_modifier",
                        extra={
                            "tau_hat_r_stage": tau_hat,
                            "tau_logit_modifier": tau_logit_modifier,
                            "r_pseudo_outcome": r_pseudo_outcome[heldout_pos],
                            "r_loss": heldout_r_loss,
                            "effect_loss": heldout_effect_loss,
                            "effect_objective": np.asarray(
                                [effect_objective] * len(heldout_pos),
                                dtype=object,
                            ),
                        },
                    )
                    return {
                        "fold": int(fold),
                        "heldout_pos": heldout_pos,
                        "tau_hat": tau_hat,
                        "tau_logit_modifier": tau_logit_modifier,
                        "r_loss": heldout_r_loss,
                        "effect_loss": heldout_effect_loss,
                        "test_tau": test_tau,
                        "attention": fold_attention,
                        "evidence": {
                            "outer_fold": int(outer_fold),
                            "inner_fold": int(fold),
                            "source_family": "htr",
                            "objective": f"effect_{effect_objective}",
                            "target_name": effect_objective,
                            "effect_objective": effect_objective,
                            "train_rows": int(len(eligible_fit_pos)),
                            "heldout_rows": int(len(heldout_pos)),
                            "outer_test_rows": int(len(test_df)),
                            "heldout_r_loss": _finite_or_none(np.mean(heldout_r_loss)),
                            "prediction_provenance": "inner_fold_model_heldout_and_outer_test",
                        },
                    }
                finally:
                    if model is not None:
                        runner._cleanup_model(model)

            def run_fold(fold: int, fit_pos: np.ndarray, heldout_pos: np.ndarray):
                store = self._active_checkpoint_store()
                if store is None:
                    return compute_fold(fold, fit_pos, heldout_pos)
                fit_pos = np.asarray(fit_pos, dtype=int)
                heldout_pos = np.asarray(heldout_pos, dtype=int)

                def validate(result: Mapping[str, Any]) -> None:
                    if len(np.asarray(result["heldout_pos"])) != len(heldout_pos):
                        raise ValueError("HTR effect checkpoint held-out row count changed")
                    for name in (
                        "tau_hat",
                        "tau_logit_modifier",
                        "r_loss",
                        "effect_loss",
                    ):
                        if len(np.asarray(result[name])) != len(heldout_pos):
                            raise ValueError(f"HTR effect checkpoint {name} row count changed")
                    if len(np.asarray(result["test_tau"])) != len(test_df):
                        raise ValueError("HTR effect checkpoint outer-test row count changed")

                return store.run(
                    f"htr/effect/{effect_objective}/fold_{int(fold):03d}",
                    lambda: compute_fold(fold, fit_pos, heldout_pos),
                    parameters={
                        "outer_fold": int(outer_fold),
                        "inner_fold": int(fold),
                        "total_folds": int(folds),
                        "effect_objective": effect_objective,
                    },
                    dependencies=("assembly/nuisance_ensemble",),
                    rows={
                        "fit": row_id_identity(
                            train_df.iloc[fit_pos]["_oci_row_id"].to_numpy()
                        ),
                        "validation": row_id_identity(
                            train_df.iloc[heldout_pos]["_oci_row_id"].to_numpy()
                        ),
                        "heldout": row_id_identity(test_df["_oci_row_id"].to_numpy()),
                    },
                    validator=validate,
                    deterministic_seed=True,
                ).value

            store = self._active_checkpoint_store()
            n_jobs = 1 if store is not None else runner._fold_n_jobs(folds)
            fold_results = _run_crossfit_fold_tasks(
                run_fold,
                split_items,
                n_jobs,
                device_context=runner._device_context_for_inner_fold,
            )
            attention_rows: List[Dict[str, Any]] = []
            test_tau_predictions = []
            inner_model_rows = []
            for result in fold_results:
                heldout_pos = result["heldout_pos"]
                r_df.loc[heldout_pos, "tau_hat_r_stage"] = result["tau_hat"]
                r_df.loc[heldout_pos, "tau_logit_modifier"] = result["tau_logit_modifier"]
                r_df.loc[heldout_pos, "r_loss"] = result["r_loss"]
                r_df.loc[heldout_pos, "effect_loss"] = result["effect_loss"]
                r_df.loc[heldout_pos, "effect_fold"] = result["fold"]
                attention_rows.extend(result["attention"])
                test_tau_predictions.append(np.asarray(result["test_tau"], dtype=float))
                inner_model_rows.append(result["evidence"])
            test_predictions = pd.DataFrame(
                {
                    "_oci_row_id": test_df["_oci_row_id"].to_numpy(),
                    "outer_fold": int(outer_fold),
                    "tau_hat_r_stage": np.nanmean(np.vstack(test_tau_predictions), axis=0),
                    "model_family": "htr",
                    "view_name": f"htr_effect_{effect_objective}",
                    "target_source": "ensemble_mean_nuisance_inner_ensemble",
                    "effect_objective": effect_objective,
                }
            )
            return {
                "train": {"predictions": r_df, "attention": attention_rows},
                "test_predictions": test_predictions,
                "inner_model_rows": inner_model_rows,
            }

    def fit_nuisance_full_predict(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        outer_fold: int,
    ) -> pd.DataFrame:
        runner = self._ensure_runner(train_df)
        model = None
        try:
            model = _NuisanceNet(
                extractor=runner._create_extractor(),
                hidden_dim=getattr(
                    runner.config.architecture,
                    "htr_prediction_head_hidden_dim",
                    64,
                ),
                outcome_type=runner.config.outcome_type,
            ).to(runner.device)
            positions = np.arange(len(train_df), dtype=int)
            runner._train_nuisance_model(
                model,
                train_df,
                positions,
                outer_fold=outer_fold,
                fold=0,
                total_folds=1,
            )
            e_hat, m_hat = runner._predict_nuisance_model(model, test_df)
        finally:
            if model is not None:
                runner._cleanup_model(model)
        return pd.DataFrame(
            {
                "_oci_row_id": test_df["_oci_row_id"].to_numpy(),
                "outer_fold": int(outer_fold),
                "e_hat": e_hat,
                "m_hat": m_hat,
                "model_family": "htr",
                "view_name": "htr_nuisance",
                "target_source": "htr_nuisance_outer_train_fit",
            }
        )

    def fit_effect_full_predict(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        nuisance_predictions: pd.DataFrame,
        outer_fold: int,
        *,
        effect_objective: str,
    ) -> pd.DataFrame:
        runner = self._ensure_runner(train_df)
        model = None
        r_df = train_df[["_oci_row_id"]].merge(
            nuisance_predictions.copy(),
            on="_oci_row_id",
            how="left",
            sort=False,
        )
        e = r_df["e_hat"].to_numpy(dtype=float)
        m = r_df["m_hat"].to_numpy(dtype=float)
        y = train_df[runner.config.outcome_column].to_numpy(dtype=float)
        t = train_df[runner.config.treatment_column].to_numpy(dtype=float)
        e_clipped = np.clip(e, runner.avf_config.e_clip, 1.0 - runner.avf_config.e_clip)
        m_clipped = clip_probability(m)
        t_resid = t - e_clipped
        y_resid = y - m
        try:
            model = _EffectNet(
                extractor=runner._create_extractor(),
                hidden_dim=getattr(
                    runner.config.architecture,
                    "htr_prediction_head_hidden_dim",
                    64,
                ),
            ).to(runner.device)
            positions = np.arange(len(train_df), dtype=int)
            with self._temporary_effect_objective(effect_objective):
                runner._train_effect_model(
                    model,
                    train_df,
                    positions,
                    y,
                    t,
                    e_clipped,
                    m_clipped,
                    y_resid,
                    t_resid,
                    outer_fold=outer_fold,
                    fold=0,
                    total_folds=1,
                )
                tau_hat = runner._predict_effect_model(model, test_df)
        finally:
            if model is not None:
                runner._cleanup_model(model)
        return pd.DataFrame(
            {
                "_oci_row_id": test_df["_oci_row_id"].to_numpy(),
                "outer_fold": int(outer_fold),
                "tau_hat_r_stage": tau_hat,
                "model_family": "htr",
                "view_name": f"htr_effect_{effect_objective}",
                "target_source": "ensemble_mean_nuisance_outer_train_fit",
                "effect_objective": effect_objective,
            }
        )


class MultiModelForestStage1Runner:
    """Primary non-agentic text-model W/X causal-forest runner."""

    def __init__(
        self,
        dataset: pd.DataFrame,
        config: AppliedInferenceConfig,
        output_path: Path,
        device: Optional[Any] = None,
        gpu_ids: Optional[Sequence[int]] = None,
        num_workers: int = 1,
        embedding_provider: Optional[Any] = None,
        htr_evidence_provider: Optional[Any] = None,
        bow_native_capture_sink: Optional[Any] = None,
        htr_native_capture_sink: Optional[Any] = None,
        matched_pair_native_capture_sink: Optional[Any] = None,
        checkpoint_store: Optional[Stage1CheckpointStore] = None,
    ) -> None:
        oracle_columns = [
            column
            for column in dataset.columns
            if str(column).lower().startswith(("true_", "oracle_", "ground_truth"))
        ]
        self.dataset = (
            dataset.drop(columns=oracle_columns, errors="ignore").reset_index(drop=True).copy()
        )
        self.dataset["_oci_row_id"] = np.arange(len(self.dataset), dtype=int)
        self.config = config
        self.output_path = Path(output_path)
        self.artifact_dir = self.output_path.parent / "stage1_text_models"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device or "cpu")
        self.gpu_ids = list(gpu_ids) if gpu_ids is not None else None
        self.num_workers = 1 if num_workers is None else int(num_workers)
        self.embedding_provider = embedding_provider
        self.htr_evidence_provider = htr_evidence_provider
        self.bow_native_capture_sink = bow_native_capture_sink
        self.htr_native_capture_sink = htr_native_capture_sink
        self.matched_pair_native_capture_sink = matched_pair_native_capture_sink
        self.checkpoint_store = checkpoint_store
        self.nn_config: MultiModelForestConfig = getattr(
            config.architecture,
            "multi_model_forest",
            MultiModelForestConfig(),
        )
        self.search_config: AgenticFeatureSearchConfig = getattr(
            config.architecture,
            "agentic_feature_search",
            AgenticFeatureSearchConfig(),
        )
        self._sync_htr_fold_parallelism()
        self.cf_config: ExplicitFeatureForestConfig = getattr(
            config.architecture,
            "explicit_feature_forest",
            ExplicitFeatureForestConfig(),
        )
        self.embedding_evidence_generator: Optional[EmbeddingContrastEvidenceGenerator] = None
        self._default_htr_provider: Optional[MultiModelForestStage1HTRProvider] = None

        self.prediction_results: Optional[pd.DataFrame] = None
        self.outer_metric_rows: List[Dict[str, Any]] = []
        self.split_provenance_rows: List[Dict[str, Any]] = []
        self.feature_manifest_rows: List[Dict[str, Any]] = []
        self.source_prediction_frames: List[pd.DataFrame] = []
        self.embedding_feature_rows: List[Dict[str, Any]] = []
        self.agentic_handoff_rows: List[Dict[str, Any]] = []
        self.inner_model_evidence_rows: List[Dict[str, Any]] = []

    def _active_checkpoint_store(self) -> Optional[Stage1CheckpointStore]:
        """Return the store only when replay cannot bypass native proof capture."""

        if self.checkpoint_store is None:
            return None
        if any(
            sink is not None
            for sink in (
                self.bow_native_capture_sink,
                self.htr_native_capture_sink,
                self.matched_pair_native_capture_sink,
            )
        ):
            return None
        if self.embedding_provider is not None or self.htr_evidence_provider is not None:
            return None
        return self.checkpoint_store

    @staticmethod
    def _checkpoint_rows(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        return {
            "fit": row_id_identity(train_df["_oci_row_id"].to_numpy()),
            "heldout": row_id_identity(test_df["_oci_row_id"].to_numpy()),
        }

    def _checkpointed_value(
        self,
        unit: str,
        compute: Callable[[], _CheckpointValue],
        *,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        parameters: Optional[Mapping[str, Any]] = None,
        dependencies: Sequence[str] = (),
        validator: Optional[Callable[[_CheckpointValue], None]] = None,
        deterministic_seed: bool = False,
    ) -> _CheckpointValue:
        store = self._active_checkpoint_store()
        if store is None:
            value = compute()
            if validator is not None:
                validator(value)
            return value
        return store.run(
            unit,
            compute,
            parameters=parameters,
            dependencies=dependencies,
            rows=self._checkpoint_rows(train_df, test_df),
            validator=validator,
            deterministic_seed=deterministic_seed,
        ).value

    def run(self) -> None:
        logger.info("=" * 80)
        logger.info("MULTI-MODEL FOREST STAGE 1 TEXT MODELS")
        logger.info("=" * 80)
        splits = self._analysis_splits()
        self.split_provenance_rows = self._split_provenance_rows(splits)
        if self._embedding_contrast_enabled() and self.embedding_provider is None:
            self._embedding_generator().prepare(self.dataset)

        outer_n_jobs = self._outer_n_jobs(len(splits))
        if outer_n_jobs > 1 and (
            self.embedding_provider is not None or self.htr_evidence_provider is not None
        ):
            logger.warning(
                "Outer fold parallelism disabled because custom embedding_provider "
                "or htr_evidence_provider objects were supplied."
            )
            outer_n_jobs = 1

        if outer_n_jobs > 1:
            outer_devices = self._outer_devices(outer_n_jobs)
            outer_backend = self._outer_backend_name()
            inner_workers = self._inner_workers_for_outer_job(outer_n_jobs)
            logger.info(
                "Running %s multi-model stage1 outer folds with "
                "outer_parallelism=%s outer_backend=%s inner_workers_per_outer=%s "
                "devices=%s bow_fold_parallelism=%s htr_fold_parallelism=%s",
                len(splits),
                outer_n_jobs,
                outer_backend,
                inner_workers,
                [str(device) for device in outer_devices],
                self._bow_fold_parallelism_setting(),
                self._htr_fold_parallelism_setting(),
            )
            if outer_backend == "threads":
                with ThreadPoolExecutor(
                    max_workers=outer_n_jobs,
                    thread_name_prefix="mm-stage1-outer",
                ) as executor:
                    futures = [
                        executor.submit(
                            self._run_one_analysis_split_isolated,
                            outer_fold=int(outer_fold),
                            train_idx=np.asarray(train_idx, dtype=int),
                            test_idx=np.asarray(test_idx, dtype=int),
                            device=outer_devices[(task_index - 1) % len(outer_devices)],
                            outer_n_jobs=outer_n_jobs,
                        )
                        for task_index, (outer_fold, train_idx, test_idx) in enumerate(
                            splits,
                            start=1,
                        )
                    ]
                    fold_results = [future.result() for future in futures]
            else:
                fold_results = Parallel(
                    n_jobs=outer_n_jobs,
                    backend="loky",
                    batch_size=1,
                    pre_dispatch="all",
                )(
                    delayed(_run_stage1_outer_fold_job)(
                        dataset=self.dataset,
                        config=self.config,
                        output_path=self.output_path,
                        outer_fold=int(outer_fold),
                        train_idx=np.asarray(train_idx, dtype=int),
                        test_idx=np.asarray(test_idx, dtype=int),
                        device=str(outer_devices[(task_index - 1) % len(outer_devices)]),
                        gpu_ids=(
                            [int(outer_devices[(task_index - 1) % len(outer_devices)].index)]
                            if outer_devices[(task_index - 1) % len(outer_devices)].type == "cuda"
                            and outer_devices[(task_index - 1) % len(outer_devices)].index
                            is not None
                            else None
                        ),
                        num_workers=inner_workers,
                        htr_dataloader_workers=0,
                    )
                    for task_index, (outer_fold, train_idx, test_idx) in enumerate(
                        splits,
                        start=1,
                    )
                )
            fold_results = sorted(fold_results, key=lambda item: item["outer_fold"])
            prediction_frames = [item["predictions"] for item in fold_results]
            for item in fold_results:
                self.feature_manifest_rows.extend(item["feature_manifest_rows"])
                self.source_prediction_frames.extend(item["source_prediction_frames"])
                self.embedding_feature_rows.extend(item["embedding_feature_rows"])
                self.outer_metric_rows.extend(item["outer_metric_rows"])
                self.agentic_handoff_rows.extend(item.get("agentic_handoff_rows", []))
                self.inner_model_evidence_rows.extend(item.get("inner_model_evidence_rows", []))
        else:
            prediction_frames = []
            for outer_fold, train_idx, test_idx in splits:
                logger.info(
                    "Multi-model stage1 fold %s: train=%s test=%s device=%s",
                    outer_fold,
                    len(train_idx),
                    len(test_idx),
                    self.device,
                )
                prediction_frames.append(
                    self._run_one_analysis_split(
                        outer_fold=int(outer_fold),
                        train_idx=np.asarray(train_idx, dtype=int),
                        test_idx=np.asarray(test_idx, dtype=int),
                    )
                )

        results_df = pd.concat(prediction_frames).sort_values("_oci_row_id")
        self.prediction_results = results_df
        self._save_outputs(results_df)

    def _run_one_analysis_split_isolated(
        self,
        *,
        outer_fold: int,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        device: torch.device,
        outer_n_jobs: int,
    ) -> Dict[str, Any]:
        logger.info(
            "Multi-model stage1 isolated fold %s: train=%s test=%s device=%s",
            outer_fold,
            len(train_idx),
            len(test_idx),
            device,
        )
        gpu_ids = None
        if device.type == "cuda" and device.index is not None:
            gpu_ids = [int(device.index)]
        return _run_stage1_outer_fold_job(
            dataset=self.dataset,
            config=self.config,
            output_path=self.output_path,
            outer_fold=outer_fold,
            train_idx=train_idx,
            test_idx=test_idx,
            device=str(device),
            gpu_ids=gpu_ids,
            num_workers=self._inner_workers_for_outer_job(outer_n_jobs),
            htr_dataloader_workers=None,
        )

    def _analysis_splits(self) -> List[Tuple[int, np.ndarray, np.ndarray]]:
        if self.config.cv_folds > 1:
            splits = KFold(
                n_splits=int(self.config.cv_folds),
                shuffle=True,
                random_state=42,
            ).split(self.dataset)
            return [
                (fold, np.asarray(train_idx), np.asarray(test_idx))
                for fold, (train_idx, test_idx) in enumerate(splits, start=1)
            ]

        split_col = self.config.split_column
        if split_col in self.dataset.columns and "test" in set(self.dataset[split_col]):
            train_mask = self.dataset[split_col].isin(["train", "val"])
            test_mask = self.dataset[split_col] == "test"
            return [
                (
                    1,
                    np.where(train_mask.to_numpy())[0],
                    np.where(test_mask.to_numpy())[0],
                )
            ]

        if bool(getattr(self.nn_config, "require_honest_outer_split", False)):
            raise ValueError(
                "multi_model_forest.require_honest_outer_split=True "
                "requires cv_folds > 1 or split_column with a 'test' split"
            )
        all_idx = np.arange(len(self.dataset), dtype=int)
        logger.warning(
            "No held-out split configured for multi_model_forest Stage 1; "
            "predictions will be labeled full_data_refit_non_honest."
        )
        return [(1, all_idx, all_idx)]

    def build_discovery_handoff_row(
        self,
        *,
        train_idx: np.ndarray,
        heldout_idx: np.ndarray,
        fold_key: int,
        outer_fold: int,
        scope: str,
        inner_fold: Optional[int] = None,
        heldout_rows: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Fit every Stage 1 text architecture for one exact handoff context.

        This uses the same feature-bundle producer as the primary forest.  The
        handoff-only research path therefore cannot fall back to the older
        discovery implementation that predates matched-pair uplift.  Evidence
        is fitted only from ``train_idx``; held-out rows are prediction targets.
        """

        train_idx = np.asarray(train_idx, dtype=int)
        heldout_idx = np.asarray(heldout_idx, dtype=int)
        if train_idx.ndim != 1 or heldout_idx.ndim != 1:
            raise ValueError("handoff context row indices must be one-dimensional")
        if not len(train_idx) or not len(heldout_idx):
            raise ValueError("handoff context requires nonempty fit and held-out rows")
        if np.intersect1d(train_idx, heldout_idx).size:
            raise ValueError("handoff context fit and held-out rows must be disjoint")
        if (
            np.any(train_idx < 0)
            or np.any(heldout_idx < 0)
            or np.any(train_idx >= len(self.dataset))
            or np.any(heldout_idx >= len(self.dataset))
        ):
            raise IndexError("handoff context row index is outside the dataset")

        train_df = self.dataset.iloc[train_idx].reset_index(drop=True)
        heldout_df = self.dataset.iloc[heldout_idx].reset_index(drop=True)
        bundle = self._build_feature_bundle(
            train_df=train_df,
            test_df=heldout_df,
            outer_fold=int(fold_key),
        )
        if bundle.prediction_frames:
            predictions = pd.concat(bundle.prediction_frames, ignore_index=True)
            predictions["fold_key"] = predictions["outer_fold"].astype(int)
            predictions["outer_fold"] = int(outer_fold)
            predictions["inner_fold"] = (
                None if inner_fold is None else int(inner_fold)
            )
            predictions["scope"] = str(scope)
            predictions["architecture"] = predictions["source_name"].map(
                _source_prediction_architecture
            )
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            predictions.to_parquet(self.output_path, index=False)
        result = copy.deepcopy(bundle.handoff_evidence or {})
        metrics = dict(result.get("metrics") or {})
        metrics.update(bundle.metrics)
        metrics["handoff_fold_key"] = int(fold_key)
        metrics["handoff_fit_rows"] = int(len(train_df))
        metrics["handoff_heldout_rows"] = int(len(heldout_df))
        result["metrics"] = metrics
        result["context"] = self._build_primary_agent_context(
            outer_fold=int(outer_fold),
            discovery_df=train_df,
            metrics=metrics,
            importance=result.get("importance") or {},
            embedding_evidence=result.get("embedding_contrast_evidence") or {},
            htr_evidence=result.get("htr_evidence") or {},
        )
        context_payload = result["context"]
        if isinstance(context_payload, dict):
            context_payload.update(
                {
                    "fold_key": int(fold_key),
                    "consistency_scope": str(scope),
                    "inner_fold": None if inner_fold is None else int(inner_fold),
                    "fit_rows": int(len(train_df)),
                    "heldout_rows": int(len(heldout_df)),
                }
            )
            provenance = dict(context_payload.get("handoff_provenance") or {})
            provenance.update(
                {
                    "source": "multi_model_forest_stage1_exact_context",
                    "exact_context_refit": True,
                    "stage2_raw_text_modeling_required": False,
                }
            )
            context_payload["handoff_provenance"] = provenance
        return _agentic_discovery_handoff_row(
            result,
            fold_key=int(fold_key),
            outer_fold=int(outer_fold),
            scope=str(scope),
            n_rows=len(train_df),
            inner_fold=inner_fold,
            heldout_rows=(len(heldout_df) if heldout_rows is None else int(heldout_rows)),
        )

    def _split_provenance_rows(
        self,
        splits: Sequence[Tuple[int, np.ndarray, np.ndarray]],
    ) -> List[Dict[str, Any]]:
        rows = []
        for outer_fold, train_idx, test_idx in splits:
            honest = _split_is_honest(train_idx, test_idx)
            rows.append(
                {
                    "outer_fold": int(outer_fold),
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "honest_outer_holdout": bool(honest),
                    "estimation_provenance": (
                        "honest_outer_fold" if honest else "full_data_refit_non_honest"
                    ),
                }
            )
        return rows

    def _run_one_analysis_split(
        self,
        *,
        outer_fold: int,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
    ) -> pd.DataFrame:
        train_df = self.dataset.iloc[train_idx].reset_index(drop=True)
        test_df = self.dataset.iloc[test_idx].reset_index(drop=True)
        bundle = self._build_feature_bundle(
            train_df=train_df,
            test_df=test_df,
            outer_fold=outer_fold,
        )
        self.feature_manifest_rows.extend(bundle.feature_rows)
        self.embedding_feature_rows.extend(bundle.embedding_rows)
        self.source_prediction_frames.extend(bundle.prediction_frames)
        self.inner_model_evidence_rows.extend(bundle.inner_model_rows)

        x_train, x_test = _clean_train_test_matrices(bundle.x_train, bundle.x_test)
        w_train, w_test = _clean_train_test_matrices(bundle.w_train, bundle.w_test)
        if x_train.shape[1] == 0:
            x_train = np.zeros((len(train_df), 1), dtype=np.float32)
            x_test = np.zeros((len(test_df), 1), dtype=np.float32)
            bundle.x_names.append("intercept_effect")
            bundle.feature_rows.append(
                {
                    "outer_fold": int(outer_fold),
                    "feature_name": "intercept_effect",
                    "feature_role": "X",
                    "source_family": "intercept",
                    "provenance": "fallback_no_effect_features",
                }
            )
        if w_train.shape[1] == 0:
            w_train = None
            w_test = None

        t_train = train_df[self.config.treatment_column].to_numpy(dtype=float)
        y_train = train_df[self.config.outcome_column].to_numpy(dtype=float)
        t_test = test_df[self.config.treatment_column].to_numpy(dtype=float)
        y_test = test_df[self.config.outcome_column].to_numpy(dtype=float)

        forest = CausalForestHead(
            n_estimators=self.cf_config.n_estimators,
            max_depth=self.cf_config.max_depth,
            min_samples_leaf=self.cf_config.min_samples_leaf,
            max_features=self.cf_config.max_features,
            honest=self.cf_config.honest,
            inference=self.cf_config.inference,
            random_state=42 + int(outer_fold),
            outcome_type=self.config.outcome_type,
        )
        forest.fit(X=x_train, T=t_train, Y=y_train, W=w_train)
        cf_preds = forest.predict(x_test, return_ci=True)
        tau = cf_preds["tau_pred"]

        nuisance_train = _hstack_present(x_train, w_train)
        nuisance_test = _hstack_present(x_test, w_test)
        if nuisance_train is None or nuisance_test is None:
            raise ValueError("Unable to build nuisance matrices for final predictions")
        propensity = _fit_predict_propensity(
            nuisance_train,
            t_train,
            nuisance_test,
            self.cf_config,
            random_state=142 + int(outer_fold),
        )
        outcome_pred = _fit_predict_outcome(
            nuisance_train,
            y_train,
            nuisance_test,
            self.config.outcome_type,
            self.cf_config,
            random_state=242 + int(outer_fold),
        )
        y0_prob = outcome_pred - propensity * tau
        y1_prob = outcome_pred + (1.0 - propensity) * tau
        if str(self.config.outcome_type).lower() == "binary":
            y0_prob = np.clip(y0_prob, 0.0, 1.0)
            y1_prob = np.clip(y1_prob, 0.0, 1.0)

        predictions = test_df.copy()
        honest = _split_is_honest(train_idx, test_idx)
        predictions["pred_ite_prob"] = tau
        predictions["pred_y0_prob"] = y0_prob
        predictions["pred_y1_prob"] = y1_prob
        predictions["pred_propensity_prob"] = propensity
        predictions["pred_outcome_prob"] = outcome_pred
        predictions["cv_fold"] = int(outer_fold)
        predictions["outer_fold"] = int(outer_fold)
        predictions["honest_outer_holdout"] = bool(honest)
        predictions["estimation_provenance"] = (
            "honest_outer_fold" if honest else "full_data_refit_non_honest"
        )
        predictions["selected_feature_names"] = ",".join(bundle.x_names + bundle.w_names)
        predictions["selected_feature_roles"] = json.dumps(
            {"X": bundle.x_names, "W": bundle.w_names}
        )
        predictions["selected_confounder_names"] = ",".join(bundle.w_names)
        predictions["selected_effect_modifier_names"] = ",".join(bundle.x_names)
        if "tau_lower" in cf_preds:
            predictions["pred_ite_lower"] = cf_preds["tau_lower"]
            predictions["pred_ite_upper"] = cf_preds["tau_upper"]

        metrics = {
            "outer_fold": int(outer_fold),
            "honest_outer_holdout": bool(honest),
            "estimation_provenance": (
                "honest_outer_fold" if honest else "full_data_refit_non_honest"
            ),
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "n_x_features": int(x_train.shape[1]),
            "n_w_features": 0 if w_train is None else int(w_train.shape[1]),
            "x_feature_names": bundle.x_names,
            "w_feature_names": bundle.w_names,
            "ate_estimate": float(np.mean(tau)),
            "r_loss": _r_loss(y_test, t_test, outcome_pred, propensity, tau),
            "treatment_auroc": _safe_roc_auc(t_test, propensity),
            "feature_discovery_methods": self._enabled_feature_discovery_methods(),
            **bundle.metrics,
        }
        if str(self.config.outcome_type).lower() == "continuous":
            metrics["outcome_rmse"] = float(np.sqrt(mean_squared_error(y_test, outcome_pred)))
        else:
            metrics["outcome_auroc"] = _safe_roc_auc(y_test, outcome_pred)
        self.outer_metric_rows.append(metrics)

        if bundle.handoff_evidence is not None:
            handoff_result = copy.deepcopy(bundle.handoff_evidence)
            handoff_metrics = dict(handoff_result.get("metrics") or {})
            handoff_metrics.update(metrics)
            handoff_result["metrics"] = handoff_metrics
            handoff_result["context"] = self._build_primary_agent_context(
                outer_fold=outer_fold,
                discovery_df=train_df,
                metrics=handoff_metrics,
                importance=handoff_result.get("importance") or {},
                embedding_evidence=handoff_result.get("embedding_contrast_evidence") or {},
                htr_evidence=handoff_result.get("htr_evidence") or {},
            )
            self.agentic_handoff_rows.append(
                _agentic_discovery_handoff_row(
                    handoff_result,
                    fold_key=int(outer_fold),
                    outer_fold=int(outer_fold),
                    scope="full_outer_train",
                    n_rows=len(train_df),
                )
            )

        fold_dir = self.artifact_dir / f"outer_fold_{int(outer_fold):03d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            fold_dir / "feature_matrices.npz",
            x_train=x_train,
            x_test=x_test,
            w_train=np.zeros((len(train_df), 0), dtype=np.float32) if w_train is None else w_train,
            w_test=np.zeros((len(test_df), 0), dtype=np.float32) if w_test is None else w_test,
            x_feature_names=np.asarray(bundle.x_names, dtype=object),
            w_feature_names=np.asarray(bundle.w_names, dtype=object),
        )
        predictions.to_parquet(fold_dir / "predictions.parquet", index=False)
        return predictions

    def _build_feature_bundle(
        self,
        *,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        outer_fold: int,
    ) -> _FeatureBundle:
        def compute() -> Dict[str, Any]:
            return _feature_bundle_checkpoint_payload(
                self._build_feature_bundle_uncached(
                    train_df=train_df,
                    test_df=test_df,
                    outer_fold=outer_fold,
                )
            )

        def validate(payload: Mapping[str, Any]) -> None:
            required = {
                "x_train",
                "x_test",
                "w_train",
                "w_test",
                "x_names",
                "w_names",
                "feature_rows",
                "prediction_frames",
                "embedding_rows",
                "metrics",
                "handoff_evidence",
                "inner_model_rows",
            }
            missing = sorted(required - set(payload))
            if missing:
                raise ValueError(f"feature-bundle checkpoint is missing fields: {missing}")
            for name, expected_rows in (
                ("x_train", len(train_df)),
                ("w_train", len(train_df)),
                ("x_test", len(test_df)),
                ("w_test", len(test_df)),
            ):
                matrix = np.asarray(payload[name])
                if matrix.ndim != 2 or matrix.shape[0] != expected_rows:
                    raise ValueError(
                        f"feature-bundle checkpoint {name} has shape {matrix.shape}; "
                        f"expected ({expected_rows}, n_features)"
                    )

        payload = self._checkpointed_value(
            "assembly/feature_bundle",
            compute,
            train_df=train_df,
            test_df=test_df,
            parameters={"outer_fold": int(outer_fold), "format_version": 1},
            validator=validate,
        )
        return _feature_bundle_from_checkpoint(payload)

    def _build_feature_bundle_uncached(
        self,
        *,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        outer_fold: int,
    ) -> _FeatureBundle:
        texts_train = _normalize_texts(train_df[self.config.text_column].fillna(""))
        texts_test = _normalize_texts(test_df[self.config.text_column].fillna(""))
        y = train_df[self.config.outcome_column].to_numpy(dtype=float)
        t = train_df[self.config.treatment_column].to_numpy(dtype=float)
        x_train_cols: List[np.ndarray] = []
        x_test_cols: List[np.ndarray] = []
        w_train_cols: List[np.ndarray] = []
        w_test_cols: List[np.ndarray] = []
        x_names: List[str] = []
        w_names: List[str] = []
        feature_rows: List[Dict[str, Any]] = []
        prediction_frames: List[pd.DataFrame] = []
        embedding_rows: List[Dict[str, Any]] = []
        metrics: Dict[str, Any] = {}
        nuisance_train: List[Tuple[str, np.ndarray, np.ndarray]] = []
        nuisance_test: List[Tuple[str, np.ndarray, np.ndarray]] = []
        bow_nuisance_by_view: List[Dict[str, Any]] = []
        bow_view_results: List[Dict[str, Any]] = []
        ensemble_view_results: List[Dict[str, Any]] = []
        htr_evidence: Dict[str, Any] = {}
        inner_model_rows: List[Dict[str, Any]] = []
        nuisance_checkpoint_units: List[str] = []

        if self._bow_enabled():
            for view_index, view in enumerate(self.nn_config.bow_views):
                e_train, e_test, fold_rows = self._checkpointed_value(
                    f"bow/nuisance/view_{int(view_index):03d}_treatment",
                    lambda: self._fit_bow_binary_train_test(
                        texts_train,
                        texts_test,
                        t,
                        outer_fold=outer_fold,
                        view=view,
                        view_index=view_index,
                        label_name="treatment",
                        train_row_ids=train_df["_oci_row_id"].to_numpy(),
                        test_row_ids=test_df["_oci_row_id"].to_numpy(),
                        checkpoint_namespace=(
                            f"bow/nuisance/view_{int(view_index):03d}_treatment"
                        ),
                    ),
                    train_df=train_df,
                    test_df=test_df,
                    parameters={
                        "outer_fold": int(outer_fold),
                        "view_index": int(view_index),
                        "view": _bow_view_to_dict(view),
                        "target": "treatment",
                    },
                    validator=lambda result: _validate_train_test_prediction_tuple(
                        result,
                        len(train_df),
                        len(test_df),
                        "BoW treatment nuisance",
                    ),
                )
                nuisance_checkpoint_units.append(
                    f"bow/nuisance/view_{int(view_index):03d}_treatment"
                )
                inner_model_rows.extend(fold_rows)
                if str(self.config.outcome_type).lower() == "continuous":
                    m_train, m_test, fold_rows = self._checkpointed_value(
                        f"bow/nuisance/view_{int(view_index):03d}_outcome",
                        lambda: self._fit_bow_regression_train_test(
                            texts_train,
                            texts_test,
                            y,
                            None,
                            outer_fold=outer_fold,
                            view=view,
                            view_index=view_index,
                            target_name="outcome",
                            train_row_ids=train_df["_oci_row_id"].to_numpy(),
                            test_row_ids=test_df["_oci_row_id"].to_numpy(),
                            checkpoint_namespace=(
                                f"bow/nuisance/view_{int(view_index):03d}_outcome"
                            ),
                        ),
                        train_df=train_df,
                        test_df=test_df,
                        parameters={
                            "outer_fold": int(outer_fold),
                            "view_index": int(view_index),
                            "view": _bow_view_to_dict(view),
                            "target": "continuous_outcome",
                        },
                        validator=lambda result: _validate_train_test_prediction_tuple(
                            result,
                            len(train_df),
                            len(test_df),
                            "BoW outcome nuisance",
                        ),
                    )
                    nuisance_checkpoint_units.append(
                        f"bow/nuisance/view_{int(view_index):03d}_outcome"
                    )
                    inner_model_rows.extend(fold_rows)
                else:
                    m_train, m_test, fold_rows = self._checkpointed_value(
                        f"bow/nuisance/view_{int(view_index):03d}_outcome",
                        lambda: self._fit_bow_binary_train_test(
                            texts_train,
                            texts_test,
                            y,
                            outer_fold=outer_fold,
                            view=view,
                            view_index=view_index,
                            label_name="outcome",
                            train_row_ids=train_df["_oci_row_id"].to_numpy(),
                            test_row_ids=test_df["_oci_row_id"].to_numpy(),
                            checkpoint_namespace=(
                                f"bow/nuisance/view_{int(view_index):03d}_outcome"
                            ),
                        ),
                        train_df=train_df,
                        test_df=test_df,
                        parameters={
                            "outer_fold": int(outer_fold),
                            "view_index": int(view_index),
                            "view": _bow_view_to_dict(view),
                            "target": "binary_outcome",
                        },
                        validator=lambda result: _validate_train_test_prediction_tuple(
                            result,
                            len(train_df),
                            len(test_df),
                            "BoW outcome nuisance",
                        ),
                    )
                    nuisance_checkpoint_units.append(
                        f"bow/nuisance/view_{int(view_index):03d}_outcome"
                    )
                    inner_model_rows.extend(fold_rows)
                nuisance_train.append((view.name, e_train, m_train))
                nuisance_test.append((view.name, e_test, m_test))
                bow_nuisance_by_view.append(
                    {
                        "view": view,
                        "view_index": int(view_index),
                        "e_hat": e_train,
                        "m_hat": m_train,
                        "e_test": e_test,
                        "m_test": m_test,
                    }
                )
                _append_feature(
                    w_train_cols,
                    w_test_cols,
                    w_names,
                    feature_rows,
                    train=e_train,
                    test=e_test,
                    name=f"bow__{view.name}__treatment_pred",
                    role="W",
                    source_family="bow",
                    outer_fold=outer_fold,
                    objective="treatment_nuisance",
                    provenance="inner_oof_train_outer_train_fit_test",
                    view_config=_bow_view_to_dict(view),
                )
                _append_feature(
                    w_train_cols,
                    w_test_cols,
                    w_names,
                    feature_rows,
                    train=m_train,
                    test=m_test,
                    name=f"bow__{view.name}__outcome_pred",
                    role="W",
                    source_family="bow",
                    outer_fold=outer_fold,
                    objective="outcome_nuisance",
                    provenance="inner_oof_train_outer_train_fit_test",
                    view_config=_bow_view_to_dict(view),
                )
                prediction_frames.append(
                    _source_prediction_frame(
                        train_df,
                        test_df,
                        outer_fold=outer_fold,
                        source_name=f"bow__{view.name}__nuisance",
                        values={
                            "e_hat": (e_train, e_test),
                            "m_hat": (m_train, m_test),
                        },
                    )
                )

        htr_train_result = None
        htr_test_predictions = None
        if self._htr_enabled():
            htr_provider = self._htr_provider()
            if hasattr(htr_provider, "fit_nuisance_inner_ensemble_predict"):
                htr_bundle = htr_provider.fit_nuisance_inner_ensemble_predict(
                    train_df,
                    test_df,
                    outer_fold,
                )
                htr_train_result = htr_bundle["train"]
                htr_test_predictions = htr_bundle["test_predictions"]
                inner_model_rows.extend(htr_bundle.get("inner_model_rows", []))
                htr_nuisance_folds = _bounded_fold_count(
                    self.config.architecture.agentic_attention_variable_forest.nuisance_folds,
                    len(train_df),
                )
                nuisance_checkpoint_units.extend(
                    f"htr/nuisance/fold_{fold:03d}"
                    for fold in range(1, htr_nuisance_folds + 1)
                )
            else:
                htr_train_result = htr_provider.fit_nuisance(train_df, outer_fold)
                htr_test_predictions = htr_provider.fit_nuisance_full_predict(
                    train_df,
                    test_df,
                    outer_fold,
                )
            htr_train_predictions = _align_htr_prediction_frame(
                htr_train_result.get("predictions"),
                train_df,
                required_columns=["e_hat", "m_hat"],
                source="htr_nuisance",
            )
            htr_test_predictions = _align_htr_prediction_frame(
                htr_test_predictions,
                test_df,
                required_columns=["e_hat", "m_hat"],
                source="htr_nuisance_outer_train_fit",
            )
            htr_e_train = htr_train_predictions["e_hat"].to_numpy(dtype=float)
            htr_m_train = htr_train_predictions["m_hat"].to_numpy(dtype=float)
            htr_e_test = htr_test_predictions["e_hat"].to_numpy(dtype=float)
            htr_m_test = htr_test_predictions["m_hat"].to_numpy(dtype=float)
            if self.htr_native_capture_sink is not None:
                for name, values, role in (
                    ("htr_e_fit", htr_e_train, "fit_nuisance"),
                    ("htr_m_fit", htr_m_train, "fit_nuisance"),
                    ("htr_e_heldout", htr_e_test, "heldout_nuisance"),
                    ("htr_m_heldout", htr_m_test, "heldout_nuisance"),
                ):
                    self.htr_native_capture_sink.record_scope_output(
                        name,
                        values,
                        role=role,
                    )
            nuisance_train.append(("htr_nuisance", htr_e_train, htr_m_train))
            nuisance_test.append(("htr_nuisance", htr_e_test, htr_m_test))
            _append_feature(
                w_train_cols,
                w_test_cols,
                w_names,
                feature_rows,
                train=htr_e_train,
                test=htr_e_test,
                name="htr__nuisance__treatment_pred",
                role="W",
                source_family="htr",
                outer_fold=outer_fold,
                objective="treatment_nuisance",
                provenance="inner_oof_train_outer_train_fit_test",
            )
            _append_feature(
                w_train_cols,
                w_test_cols,
                w_names,
                feature_rows,
                train=htr_m_train,
                test=htr_m_test,
                name="htr__nuisance__outcome_pred",
                role="W",
                source_family="htr",
                outer_fold=outer_fold,
                objective="outcome_nuisance",
                provenance="inner_oof_train_outer_train_fit_test",
            )
            metrics["htr_treatment_auroc"] = _safe_roc_auc(t, htr_e_train)
            if str(self.config.outcome_type).lower() == "continuous":
                metrics["htr_outcome_rmse"] = float(np.sqrt(mean_squared_error(y, htr_m_train)))
            else:
                metrics["htr_outcome_auroc"] = _safe_roc_auc(y, htr_m_train)
            htr_attention = [dict(row) for row in htr_train_result.get("attention", []) or []]
            for row in htr_attention:
                row.setdefault("model_family", "htr")
                row.setdefault("target_source", "htr_nuisance")
            htr_evidence["nuisance"] = {
                "metrics": _htr_nuisance_metrics(
                    discovery_df=train_df,
                    predictions=htr_train_predictions,
                    treatment_column=self.config.treatment_column,
                    outcome_column=self.config.outcome_column,
                    outcome_type=self.config.outcome_type,
                ),
                "attention": htr_attention,
            }
            prediction_frames.append(
                _source_prediction_frame(
                    train_df,
                    test_df,
                    outer_fold=outer_fold,
                    source_name="htr__nuisance",
                    values={
                        "e_hat": (htr_e_train, htr_e_test),
                        "m_hat": (htr_m_train, htr_m_test),
                    },
                )
            )

        if not nuisance_train:
            raise ValueError("multi_model_forest Stage 1 requires at least one nuisance source")
        e_train = np.nanmean(np.vstack([item[1] for item in nuisance_train]), axis=0)
        m_train = np.nanmean(np.vstack([item[2] for item in nuisance_train]), axis=0)
        e_test = np.nanmean(np.vstack([item[1] for item in nuisance_test]), axis=0)
        m_test = np.nanmean(np.vstack([item[2] for item in nuisance_test]), axis=0)
        e_train_clip = np.clip(e_train, self.nn_config.e_clip, 1.0 - self.nn_config.e_clip)
        t_resid = t - e_train_clip
        y_resid = y - m_train
        pseudo_target = y_resid / t_resid
        r_weight = np.square(t_resid)
        metrics.update(
            {
                "n_nuisance_sources": int(len(nuisance_train)),
                "nuisance_sources": [item[0] for item in nuisance_train],
                "ensemble_treatment_auroc": _safe_roc_auc(t, e_train),
                "ensemble_pseudo_target_mean": _finite_or_none(np.mean(pseudo_target)),
                "ensemble_pseudo_target_std": _finite_or_none(np.std(pseudo_target)),
            }
        )
        if str(self.config.outcome_type).lower() == "continuous":
            metrics["ensemble_outcome_rmse"] = float(np.sqrt(mean_squared_error(y, m_train)))
        else:
            metrics["ensemble_outcome_auroc"] = _safe_roc_auc(y, m_train)
        ensemble_nuisance_train = pd.DataFrame(
            {
                "_oci_row_id": train_df["_oci_row_id"].to_numpy(),
                "outer_fold": int(outer_fold),
                "e_hat": e_train,
                "m_hat": m_train,
                "y_residual": y_resid,
                "t_residual": t_resid,
                "r_pseudo_outcome": pseudo_target,
                "pseudo_target": pseudo_target,
                "r_loss_at_zero_tau": np.square(y_resid),
                "target_source": "ensemble_mean_nuisance",
            }
        )
        ensemble_nuisance_test = pd.DataFrame(
            {
                "_oci_row_id": test_df["_oci_row_id"].to_numpy(),
                "outer_fold": int(outer_fold),
                "e_hat": e_test,
                "m_hat": m_test,
                "target_source": "ensemble_mean_nuisance_inner_ensemble",
            }
        )
        ensemble_payload = self._checkpointed_value(
            "assembly/nuisance_ensemble",
            lambda: {
                "e_train": e_train,
                "m_train": m_train,
                "e_test": e_test,
                "m_test": m_test,
                "e_train_clip": e_train_clip,
                "t_resid": t_resid,
                "y_resid": y_resid,
                "pseudo_target": pseudo_target,
                "r_weight": r_weight,
                "train_predictions": ensemble_nuisance_train,
                "test_predictions": ensemble_nuisance_test,
            },
            train_df=train_df,
            test_df=test_df,
            parameters={
                "outer_fold": int(outer_fold),
                "sources": [item[0] for item in nuisance_train],
                "aggregation": "unweighted_nanmean_v1",
            },
            dependencies=tuple(nuisance_checkpoint_units),
            validator=lambda payload: _validate_nuisance_ensemble_payload(
                payload,
                len(train_df),
                len(test_df),
            ),
        )
        e_train = np.asarray(ensemble_payload["e_train"], dtype=float)
        m_train = np.asarray(ensemble_payload["m_train"], dtype=float)
        e_test = np.asarray(ensemble_payload["e_test"], dtype=float)
        m_test = np.asarray(ensemble_payload["m_test"], dtype=float)
        e_train_clip = np.asarray(ensemble_payload["e_train_clip"], dtype=float)
        t_resid = np.asarray(ensemble_payload["t_resid"], dtype=float)
        y_resid = np.asarray(ensemble_payload["y_resid"], dtype=float)
        pseudo_target = np.asarray(ensemble_payload["pseudo_target"], dtype=float)
        r_weight = np.asarray(ensemble_payload["r_weight"], dtype=float)
        ensemble_nuisance_train = ensemble_payload["train_predictions"]
        ensemble_nuisance_test = ensemble_payload["test_predictions"]
        if self.bow_native_capture_sink is not None:
            self.bow_native_capture_sink.record_scope_output(
                "treatment",
                t,
                role="fit_label",
            )
            self.bow_native_capture_sink.record_scope_output(
                "outcome",
                y,
                role="fit_label",
            )
            for source_index, (
                (source_name, source_e_train, source_m_train),
                (_test_name, source_e_test, source_m_test),
            ) in enumerate(zip(nuisance_train, nuisance_test)):
                if source_name != _test_name:
                    raise RuntimeError("BoW nuisance train/test source order changed")
                self.bow_native_capture_sink.record_nuisance_source(
                    source_index=source_index,
                    source_name=source_name,
                    e_fit=source_e_train,
                    m_fit=source_m_train,
                    e_heldout=source_e_test,
                    m_heldout=source_m_test,
                )
            for nuisance_view in bow_nuisance_by_view:
                view_index = int(nuisance_view["view_index"])
                self.bow_native_capture_sink.record_scope_output(
                    f"view_{view_index:04d}_e_fit",
                    nuisance_view["e_hat"],
                    role="fit_nuisance",
                )
                self.bow_native_capture_sink.record_scope_output(
                    f"view_{view_index:04d}_m_fit",
                    nuisance_view["m_hat"],
                    role="fit_nuisance",
                )
                self.bow_native_capture_sink.record_scope_output(
                    f"view_{view_index:04d}_e_heldout",
                    nuisance_view["e_test"],
                    role="heldout_nuisance",
                )
                self.bow_native_capture_sink.record_scope_output(
                    f"view_{view_index:04d}_m_heldout",
                    nuisance_view["m_test"],
                    role="heldout_nuisance",
                )
            for name, values, role in (
                ("ensemble_e_fit", e_train, "fit_nuisance"),
                ("ensemble_m_fit", m_train, "fit_nuisance"),
                ("ensemble_e_heldout", e_test, "heldout_nuisance"),
                ("ensemble_m_heldout", m_test, "heldout_nuisance"),
                ("y_residual", y_resid, "fit_residual"),
                ("t_residual", t_resid, "fit_residual"),
                ("pseudo_target", pseudo_target, "fit_pseudo_target"),
                ("r_weight", r_weight, "fit_weight"),
            ):
                self.bow_native_capture_sink.record_scope_output(name, values, role=role)
        prediction_frames.append(
            _source_prediction_frame(
                train_df,
                test_df,
                outer_fold=outer_fold,
                source_name="ensemble_mean_nuisance",
                values={
                    "e_hat": (e_train, e_test),
                    "m_hat": (m_train, m_test),
                },
            )
        )

        if self.matched_pair_native_capture_sink is not None:
            if not self._matched_pair_bow_enabled() or not self._matched_pair_htr_enabled():
                raise RuntimeError(
                    "native matched-pair proof requires both genuine BoW and HTR "
                    "matched-pair subproducers"
                )
            self.matched_pair_native_capture_sink.record_scope_inputs(
                treatment=t,
                outcome=y,
                e_fit=e_train,
                m_fit=m_train,
                e_heldout=e_test,
                m_heldout=m_test,
            )

        bow_pair_uplift_results: List[Dict[str, Any]] = []
        if self._matched_pair_bow_enabled():
            for view_index, view in enumerate(self.nn_config.bow_views):
                try:
                    pair_payload = self._checkpointed_value(
                        f"bow/pair_uplift/view_{int(view_index):03d}",
                        lambda: _pair_result_checkpoint_payload(
                            fit_bow_pair_uplift_train_test(
                                train_df=train_df,
                                test_df=test_df,
                                texts_train=texts_train,
                                texts_test=texts_test,
                                y_train=y,
                                t_train=t,
                                e_train=e_train,
                                m_train=m_train,
                                e_test=e_test,
                                m_test=m_test,
                                vectorizer_params=_vectorizer_params(view),
                                model_params=_model_params(view),
                                outer_fold=outer_fold,
                                view_name=view.name,
                                view_index=view_index,
                                effect_folds=int(self.nn_config.effect_folds),
                                propensity_caliper=float(
                                    self.nn_config.matched_pair_propensity_caliper
                                ),
                                outcome_caliper=float(
                                    self.nn_config.matched_pair_outcome_caliper
                                ),
                                max_controls_per_candidate=int(
                                    self.nn_config.matched_pair_max_controls_per_candidate
                                ),
                                nearest_fallback_controls=int(
                                    self.nn_config.matched_pair_nearest_fallback_controls
                                ),
                                l2_alpha=float(self.nn_config.matched_pair_bow_l2_alpha),
                                max_iter=int(self.nn_config.matched_pair_bow_max_iter),
                                top_n=int(self.nn_config.top_n_features),
                                native_capture_sink=self.matched_pair_native_capture_sink,
                                checkpoint_store=self._active_checkpoint_store(),
                                checkpoint_dependencies=(
                                    "assembly/nuisance_ensemble",
                                ),
                                checkpoint_namespace=(
                                    f"bow/pair_uplift/view_{int(view_index):03d}"
                                ),
                            )
                        ),
                        train_df=train_df,
                        test_df=test_df,
                        parameters={
                            "outer_fold": int(outer_fold),
                            "view_index": int(view_index),
                            "view": _bow_view_to_dict(view),
                            "objective": "matched_pair_uplift_delta_logit",
                        },
                        dependencies=("assembly/nuisance_ensemble",),
                        validator=lambda payload: _validate_pair_result_payload(
                            payload,
                            len(train_df),
                            len(test_df),
                            "BoW matched-pair uplift",
                        ),
                    )
                    pair_result = _pair_result_from_checkpoint(pair_payload)
                except Exception as exc:
                    if self.matched_pair_native_capture_sink is not None:
                        raise RuntimeError(
                            "native matched-pair BoW proof capture failed closed"
                        ) from exc
                    logger.exception(
                        "Outer fold %s BoW matched-pair uplift failed for view=%s: %s",
                        outer_fold,
                        view.name,
                        exc,
                    )
                    inner_model_rows.append(
                        {
                            "outer_fold": int(outer_fold),
                            "source_family": "bow_pair_uplift",
                            "view_name": view.name,
                            "objective": "matched_pair_uplift_delta_logit",
                            "skipped": "exception",
                            "error": str(exc),
                        }
                    )
                    continue
                inner_model_rows.extend(pair_result.evidence_rows)
                if not pair_result.prediction_frame.empty:
                    prediction_frames.append(pair_result.prediction_frame)
                _append_feature(
                    x_train_cols,
                    x_test_cols,
                    x_names,
                    feature_rows,
                    train=pair_result.train_delta_logit,
                    test=pair_result.test_delta_logit,
                    name=f"bow__{view.name}__matched_pair_uplift_delta_logit",
                    role="X",
                    source_family="bow_pair_uplift",
                    outer_fold=outer_fold,
                    objective="matched_pair_uplift_delta_logit",
                    provenance="inner_oof_pair_model_outer_test_inner_ensemble",
                    view_config=_bow_view_to_dict(view),
                )
                _append_feature(
                    x_train_cols,
                    x_test_cols,
                    x_names,
                    feature_rows,
                    train=pair_result.train_pred_prob,
                    test=pair_result.test_pred_prob,
                    name=f"bow__{view.name}__matched_pair_treated_outcome_prob",
                    role="X",
                    source_family="bow_pair_uplift",
                    outer_fold=outer_fold,
                    objective="matched_pair_treated_outcome_probability",
                    provenance="inner_oof_pair_model_outer_test_inner_ensemble",
                    view_config=_bow_view_to_dict(view),
                )
                metrics[f"bow_pair_uplift_{view.name}_treated_oof_auroc"] = pair_result.metrics.get(
                    "treated_oof", {}
                ).get("auroc")
                metrics[f"bow_pair_uplift_{view.name}_n_train_matched_pairs"] = (
                    pair_result.metrics.get("n_train_matched_pairs")
                )
                prediction_frames.append(
                    _source_prediction_frame(
                        train_df,
                        test_df,
                        outer_fold=outer_fold,
                        source_name=f"bow__{view.name}__matched_pair_uplift",
                        values={
                            "uplift_delta_logit": (
                                pair_result.train_delta_logit,
                                pair_result.test_delta_logit,
                            ),
                            "treated_outcome_prob": (
                                pair_result.train_pred_prob,
                                pair_result.test_pred_prob,
                            ),
                            "matched_control_count": (
                                pair_result.train_n_controls,
                                pair_result.test_n_controls,
                            ),
                        },
                    )
                )
                if self.matched_pair_native_capture_sink is not None:
                    for value_name, fit_value, heldout_value, role in (
                        (
                            "delta",
                            pair_result.train_delta_logit,
                            pair_result.test_delta_logit,
                            "uplift_delta_logit",
                        ),
                        (
                            "probability",
                            pair_result.train_pred_prob,
                            pair_result.test_pred_prob,
                            "treated_outcome_probability",
                        ),
                        (
                            "n_controls",
                            pair_result.train_n_controls,
                            pair_result.test_n_controls,
                            "matched_control_count",
                        ),
                    ):
                        self.matched_pair_native_capture_sink.record_scope_output(
                            f"bow_view_{view_index:04d}_{value_name}_fit",
                            fit_value,
                            role=f"fit_{role}",
                        )
                        self.matched_pair_native_capture_sink.record_scope_output(
                            f"bow_view_{view_index:04d}_{value_name}_heldout",
                            heldout_value,
                            role=f"heldout_{role}",
                        )
                bow_pair_uplift_results.append(
                    {
                        "metrics": pair_result.metrics,
                        "importance": pair_result.feature_importance,
                        "view": view,
                        "view_name": f"pair_uplift__{view.name}",
                        "view_index": int(view_index),
                    }
                )
        if self._matched_pair_bow_enabled() and not bow_pair_uplift_results:
            raise RuntimeError(
                "matched-pair uplift is enabled, but every BoW matched-pair producer failed"
            )

        htr_pair_uplift_result = None
        if self._matched_pair_htr_enabled() and self._htr_enabled():
            htr_provider = self._htr_provider()
            if hasattr(htr_provider, "fit_pair_uplift_inner_ensemble_predict"):
                try:
                    htr_pair_uplift_result = htr_provider.fit_pair_uplift_inner_ensemble_predict(
                        train_df=train_df,
                        test_df=test_df,
                        texts_train=texts_train,
                        texts_test=texts_test,
                        y_train=y,
                        t_train=t,
                        e_train=e_train,
                        m_train=m_train,
                        e_test=e_test,
                        m_test=m_test,
                        outer_fold=outer_fold,
                        propensity_caliper=float(self.nn_config.matched_pair_propensity_caliper),
                        outcome_caliper=float(self.nn_config.matched_pair_outcome_caliper),
                        max_controls_per_candidate=int(
                            self.nn_config.matched_pair_max_controls_per_candidate
                        ),
                        nearest_fallback_controls=int(
                            self.nn_config.matched_pair_nearest_fallback_controls
                        ),
                        max_attention_pairs=int(
                            self.nn_config.matched_pair_htr_attention_pairs_per_fold
                        ),
                    )
                except Exception as exc:
                    if self.matched_pair_native_capture_sink is not None:
                        raise RuntimeError(
                            "native matched-pair HTR proof capture failed closed"
                        ) from exc
                    logger.exception(
                        "Outer fold %s HTR matched-pair uplift failed: %s",
                        outer_fold,
                        exc,
                    )
                    inner_model_rows.append(
                        {
                            "outer_fold": int(outer_fold),
                            "source_family": "htr_pair_uplift",
                            "objective": "matched_pair_uplift_delta_logit",
                            "skipped": "exception",
                            "error": str(exc),
                        }
                    )
            else:
                logger.info(
                    "Skipping HTR matched-pair uplift: provider %s does not implement "
                    "fit_pair_uplift_inner_ensemble_predict",
                    type(htr_provider).__name__,
                )
            if htr_pair_uplift_result is not None:
                inner_model_rows.extend(htr_pair_uplift_result.evidence_rows)
                if not htr_pair_uplift_result.prediction_frame.empty:
                    prediction_frames.append(htr_pair_uplift_result.prediction_frame)
                _append_feature(
                    x_train_cols,
                    x_test_cols,
                    x_names,
                    feature_rows,
                    train=htr_pair_uplift_result.train_delta_logit,
                    test=htr_pair_uplift_result.test_delta_logit,
                    name="htr__matched_pair_uplift_delta_logit",
                    role="X",
                    source_family="htr_pair_uplift",
                    outer_fold=outer_fold,
                    objective="matched_pair_uplift_delta_logit",
                    provenance="inner_oof_pair_model_outer_test_inner_ensemble",
                )
                _append_feature(
                    x_train_cols,
                    x_test_cols,
                    x_names,
                    feature_rows,
                    train=htr_pair_uplift_result.train_pred_prob,
                    test=htr_pair_uplift_result.test_pred_prob,
                    name="htr__matched_pair_treated_outcome_prob",
                    role="X",
                    source_family="htr_pair_uplift",
                    outer_fold=outer_fold,
                    objective="matched_pair_treated_outcome_probability",
                    provenance="inner_oof_pair_model_outer_test_inner_ensemble",
                )
                metrics["htr_pair_uplift_treated_oof_auroc"] = htr_pair_uplift_result.metrics.get(
                    "treated_oof", {}
                ).get("auroc")
                prediction_frames.append(
                    _source_prediction_frame(
                        train_df,
                        test_df,
                        outer_fold=outer_fold,
                        source_name="htr__matched_pair_uplift",
                        values={
                            "uplift_delta_logit": (
                                htr_pair_uplift_result.train_delta_logit,
                                htr_pair_uplift_result.test_delta_logit,
                            ),
                            "treated_outcome_prob": (
                                htr_pair_uplift_result.train_pred_prob,
                                htr_pair_uplift_result.test_pred_prob,
                            ),
                            "matched_control_count": (
                                htr_pair_uplift_result.train_n_controls,
                                htr_pair_uplift_result.test_n_controls,
                            ),
                        },
                    )
                )
                htr_pair_attention = [
                    dict(row) for row in htr_pair_uplift_result.attention_rows or []
                ]
                for row in htr_pair_attention:
                    row.setdefault("model_family", "htr_pair_uplift")
                    row.setdefault("target_source", "matched_pair_uplift_delta_logit")
                htr_evidence["pair_uplift"] = {
                    "metrics": htr_pair_uplift_result.metrics,
                    "attention": htr_pair_attention,
                    "objective": "matched_pair_uplift_delta_logit",
                }
                if self.matched_pair_native_capture_sink is not None:
                    for value_name, fit_value, heldout_value, role in (
                        (
                            "delta",
                            htr_pair_uplift_result.train_delta_logit,
                            htr_pair_uplift_result.test_delta_logit,
                            "uplift_delta_logit",
                        ),
                        (
                            "probability",
                            htr_pair_uplift_result.train_pred_prob,
                            htr_pair_uplift_result.test_pred_prob,
                            "treated_outcome_probability",
                        ),
                        (
                            "n_controls",
                            htr_pair_uplift_result.train_n_controls,
                            htr_pair_uplift_result.test_n_controls,
                            "matched_control_count",
                        ),
                    ):
                        self.matched_pair_native_capture_sink.record_scope_output(
                            f"htr_{value_name}_fit",
                            fit_value,
                            role=f"fit_{role}",
                        )
                        self.matched_pair_native_capture_sink.record_scope_output(
                            f"htr_{value_name}_heldout",
                            heldout_value,
                            role=f"heldout_{role}",
                        )
            if htr_pair_uplift_result is None:
                raise RuntimeError(
                    "matched-pair HTR uplift is enabled, but its producer returned no evidence"
                )

        if self._bow_enabled():
            for view_index, view in enumerate(self.nn_config.bow_views):
                pseudo_unit = f"bow/effect/view_{int(view_index):03d}_pseudo_outcome"
                pseudo_train, pseudo_test, fold_rows = self._checkpointed_value(
                    pseudo_unit,
                    lambda: self._fit_bow_regression_train_test(
                        texts_train,
                        texts_test,
                        pseudo_target,
                        None,
                        outer_fold=outer_fold,
                        view=view,
                        view_index=view_index,
                        target_name="effect_pseudo_target",
                        seed_offset=50_000,
                        train_row_ids=train_df["_oci_row_id"].to_numpy(),
                        test_row_ids=test_df["_oci_row_id"].to_numpy(),
                        checkpoint_namespace=pseudo_unit,
                        checkpoint_dependencies=("assembly/nuisance_ensemble",),
                    ),
                    train_df=train_df,
                    test_df=test_df,
                    parameters={
                        "outer_fold": int(outer_fold),
                        "view_index": int(view_index),
                        "view": _bow_view_to_dict(view),
                        "objective": "effect_pseudo_target",
                    },
                    dependencies=("assembly/nuisance_ensemble",),
                    validator=lambda result: _validate_train_test_prediction_tuple(
                        result,
                        len(train_df),
                        len(test_df),
                        "BoW pseudo-outcome effect",
                    ),
                )
                inner_model_rows.extend(fold_rows)
                weighted_unit = f"bow/effect/view_{int(view_index):03d}_weighted_r"
                r_train, r_test, fold_rows = self._checkpointed_value(
                    weighted_unit,
                    lambda: self._fit_bow_regression_train_test(
                        texts_train,
                        texts_test,
                        pseudo_target,
                        r_weight,
                        outer_fold=outer_fold,
                        view=view,
                        view_index=view_index,
                        target_name="effect_weighted_r",
                        seed_offset=70_000,
                        train_row_ids=train_df["_oci_row_id"].to_numpy(),
                        test_row_ids=test_df["_oci_row_id"].to_numpy(),
                        checkpoint_namespace=weighted_unit,
                        checkpoint_dependencies=("assembly/nuisance_ensemble",),
                    ),
                    train_df=train_df,
                    test_df=test_df,
                    parameters={
                        "outer_fold": int(outer_fold),
                        "view_index": int(view_index),
                        "view": _bow_view_to_dict(view),
                        "objective": "effect_weighted_r",
                    },
                    dependencies=("assembly/nuisance_ensemble",),
                    validator=lambda result: _validate_train_test_prediction_tuple(
                        result,
                        len(train_df),
                        len(test_df),
                        "BoW weighted-R effect",
                    ),
                )
                inner_model_rows.extend(fold_rows)
                if self.bow_native_capture_sink is not None:
                    for name, values, role in (
                        (
                            f"view_{view_index:04d}_pseudo_fit",
                            pseudo_train,
                            "fit_effect_output",
                        ),
                        (
                            f"view_{view_index:04d}_pseudo_heldout",
                            pseudo_test,
                            "heldout_effect_output",
                        ),
                        (
                            f"view_{view_index:04d}_weighted_fit",
                            r_train,
                            "fit_effect_output",
                        ),
                        (
                            f"view_{view_index:04d}_weighted_heldout",
                            r_test,
                            "heldout_effect_output",
                        ),
                    ):
                        self.bow_native_capture_sink.record_scope_output(
                            name,
                            values,
                            role=role,
                        )
                _append_feature(
                    x_train_cols,
                    x_test_cols,
                    x_names,
                    feature_rows,
                    train=pseudo_train,
                    test=pseudo_test,
                    name=f"bow__{view.name}__effect_pseudo_target_pred",
                    role="X",
                    source_family="bow",
                    outer_fold=outer_fold,
                    objective="r_pseudo_outcome",
                    provenance="inner_oof_train_outer_train_fit_test",
                    view_config=_bow_view_to_dict(view),
                )
                _append_feature(
                    x_train_cols,
                    x_test_cols,
                    x_names,
                    feature_rows,
                    train=r_train,
                    test=r_test,
                    name=f"bow__{view.name}__effect_weighted_r_tau_pred",
                    role="X",
                    source_family="bow",
                    outer_fold=outer_fold,
                    objective="direct_weighted_r",
                    provenance="inner_oof_train_outer_train_fit_test",
                    view_config=_bow_view_to_dict(view),
                )
                prediction_frames.append(
                    _source_prediction_frame(
                        train_df,
                        test_df,
                        outer_fold=outer_fold,
                        source_name=f"bow__{view.name}__effect",
                        values={
                            "tau_hat_pseudo_target": (pseudo_train, pseudo_test),
                            "tau_hat_weighted_r": (r_train, r_test),
                        },
                    )
                )
                nuisance_view = next(
                    (
                        item
                        for item in bow_nuisance_by_view
                        if int(item["view_index"]) == int(view_index)
                    ),
                    None,
                )
                if nuisance_view is not None:
                    importance = self._checkpointed_value(
                        f"bow/importance/view_{int(view_index):03d}",
                        lambda: self._fit_primary_feature_importance_models(
                            texts=texts_train,
                            y=y,
                            t=t,
                            pseudo_target=pseudo_target,
                            pseudo_target_sample_weight=r_weight,
                            view=view,
                        ),
                        train_df=train_df,
                        test_df=test_df,
                        parameters={
                            "outer_fold": int(outer_fold),
                            "view_index": int(view_index),
                            "view": _bow_view_to_dict(view),
                            "top_n": int(self.nn_config.top_n_features),
                        },
                        dependencies=(
                            "assembly/nuisance_ensemble",
                            f"bow/nuisance/view_{int(view_index):03d}_treatment",
                            f"bow/nuisance/view_{int(view_index):03d}_outcome",
                            pseudo_unit,
                            weighted_unit,
                        ),
                        validator=lambda value: _validate_mapping_checkpoint(
                            value,
                            "BoW importance",
                        ),
                    )
                    view_metrics = self._primary_bow_metrics(
                        discovery_df=train_df,
                        y=y,
                        t=t,
                        e_hat=np.asarray(nuisance_view["e_hat"], dtype=float),
                        m_hat=np.asarray(nuisance_view["m_hat"], dtype=float),
                        pseudo_target=pseudo_target,
                        tau_hat=r_train,
                        y_resid=y_resid,
                        t_resid=t_resid,
                    )
                    bow_view_results.append(
                        {
                            "metrics": view_metrics,
                            "importance": importance,
                            "pseudo_target": pseudo_target,
                            "t_resid": t_resid,
                            "view": view,
                            "view_name": view.name,
                            "view_index": int(view_index),
                        }
                    )
                    ensemble_view_results.append(
                        {
                            "metrics": {
                                **view_metrics,
                                "target_source": "ensemble_mean_nuisance",
                            },
                            "importance": copy.deepcopy(importance),
                            "pseudo_target": pseudo_target,
                            "t_resid": t_resid,
                            "view": view,
                            "view_name": f"ensemble_r__{view.name}",
                            "view_index": int(view_index),
                        }
                    )

        if self._htr_enabled():
            htr_effect_variants: Dict[str, Any] = {}
            htr_provider = self._htr_provider()
            for effect_objective, feature_suffix in [
                ("pseudo_outcome_mse", "effect_pseudo_target_pred"),
                ("squared_r_loss", "effect_weighted_r_tau_pred"),
            ]:
                if hasattr(htr_provider, "fit_effect_variant_inner_ensemble_predict"):
                    htr_effect_bundle = htr_provider.fit_effect_variant_inner_ensemble_predict(
                        train_df,
                        test_df,
                        ensemble_nuisance_train,
                        outer_fold,
                        effect_objective=effect_objective,
                        test_nuisance_predictions=ensemble_nuisance_test,
                    )
                    htr_effect_train = htr_effect_bundle["train"]
                    test_predictions = htr_effect_bundle["test_predictions"]
                    inner_model_rows.extend(htr_effect_bundle.get("inner_model_rows", []))
                else:
                    htr_effect_train = htr_provider.fit_effect_variant(
                        train_df,
                        ensemble_nuisance_train,
                        outer_fold,
                        effect_objective=effect_objective,
                    )
                    test_predictions = htr_provider.fit_effect_full_predict(
                        train_df,
                        test_df,
                        ensemble_nuisance_train,
                        outer_fold,
                        effect_objective=effect_objective,
                    )
                train_predictions = _align_htr_prediction_frame(
                    htr_effect_train.get("predictions"),
                    train_df,
                    required_columns=["tau_hat_r_stage"],
                    source=f"htr_effect_{effect_objective}",
                )
                test_predictions = _align_htr_prediction_frame(
                    test_predictions,
                    test_df,
                    required_columns=["tau_hat_r_stage"],
                    source=f"htr_effect_{effect_objective}_outer_train_fit",
                )
                train_tau = train_predictions["tau_hat_r_stage"].to_numpy(dtype=float)
                test_tau = test_predictions["tau_hat_r_stage"].to_numpy(dtype=float)
                if self.htr_native_capture_sink is not None:
                    self.htr_native_capture_sink.record_scope_output(
                        f"effect_{effect_objective}_fit",
                        train_tau,
                        role="fit_effect_output",
                    )
                    self.htr_native_capture_sink.record_scope_output(
                        f"effect_{effect_objective}_heldout",
                        test_tau,
                        role="heldout_effect_output",
                    )
                effect_attention = [
                    dict(row) for row in htr_effect_train.get("attention", []) or []
                ]
                for row in effect_attention:
                    row.setdefault("model_family", "htr")
                    row.setdefault("target_source", "ensemble_mean_nuisance_with_htr")
                    row.setdefault("effect_objective", effect_objective)
                effect_evidence = {
                    "metrics": _htr_effect_metrics(train_predictions),
                    "attention": effect_attention,
                    "effect_objective": effect_objective,
                }
                htr_effect_variants[effect_objective] = effect_evidence
                if effect_objective == "pseudo_outcome_mse":
                    htr_evidence["effect"] = effect_evidence
                _append_feature(
                    x_train_cols,
                    x_test_cols,
                    x_names,
                    feature_rows,
                    train=train_tau,
                    test=test_tau,
                    name=f"htr__{feature_suffix}",
                    role="X",
                    source_family="htr",
                    outer_fold=outer_fold,
                    objective=(
                        "r_pseudo_outcome"
                        if effect_objective == "pseudo_outcome_mse"
                        else "direct_weighted_r"
                    ),
                    provenance="inner_oof_train_outer_train_fit_test",
                )
                prediction_frames.append(
                    _source_prediction_frame(
                        train_df,
                        test_df,
                        outer_fold=outer_fold,
                        source_name=f"htr__{feature_suffix}",
                        values={"tau_hat": (train_tau, test_tau)},
                    )
                )
            if "effect" not in htr_evidence and htr_effect_variants:
                htr_evidence["effect"] = next(iter(htr_effect_variants.values()))
            if htr_effect_variants:
                htr_evidence["effect_variants"] = htr_effect_variants

        importance: Dict[str, Any] = _multi_view_importance(
            bow_view_results,
            top_n=int(self.nn_config.top_n_features),
        )
        importance["feature_discovery_methods"] = self._enabled_feature_discovery_methods()
        if bow_pair_uplift_results:
            pair_importance = _multi_view_importance(
                bow_pair_uplift_results,
                top_n=int(self.nn_config.top_n_features),
            )
            pair_importance["target_source"] = "matched_pair_uplift_delta_logit"
            pair_importance["pair_uplift_construction"] = (
                "Observed treated/control outer-train patients are matched on "
                "honest ensemble propensity and outcome probabilities; pair models "
                "predict a delta logit added to the matched untreated outcome logit."
            )
            importance["matched_pair_uplift"] = pair_importance
        if ensemble_view_results:
            ensemble_importance = _multi_view_importance(
                ensemble_view_results,
                top_n=int(self.nn_config.top_n_features),
            )
            nuisance_source_names = [item[0] for item in nuisance_train]
            ensemble_importance["target_source"] = (
                "ensemble_mean_nuisance_with_htr"
                if any(str(name).startswith("htr") for name in nuisance_source_names)
                else "ensemble_mean_nuisance"
            )
            ensemble_importance["nuisance_sources"] = nuisance_source_names
            ensemble_importance["pseudo_target_construction"] = (
                "mean nuisance predictions across Stage 1 text models, then "
                "(Y - mean_m_hat) / (T - mean_e_hat)"
            )
            importance["ensemble_r"] = ensemble_importance

        embedding_evidence: Dict[str, Any] = {}
        if self._embedding_contrast_enabled():
            emb = self._checkpointed_value(
                "embedding/contrast_bundle",
                lambda: self._embedding_feature_bundle(
                    train_df=train_df,
                    test_df=test_df,
                    y=y,
                    t=t,
                    pseudo_target=pseudo_target,
                    t_resid=t_resid,
                    outer_fold=outer_fold,
                ),
                train_df=train_df,
                test_df=test_df,
                parameters={
                    "outer_fold": int(outer_fold),
                    "objective": "embedding_contrast_feature_bundle",
                },
                dependencies=("assembly/nuisance_ensemble",),
                validator=lambda payload: _validate_embedding_checkpoint_payload(
                    payload,
                    len(train_df),
                    len(test_df),
                ),
            )
            for item in emb["w_features"]:
                _append_feature(
                    w_train_cols,
                    w_test_cols,
                    w_names,
                    feature_rows,
                    train=item["train"],
                    test=item["test"],
                    name=item["name"],
                    role="W",
                    source_family="embedding_contrast",
                    outer_fold=outer_fold,
                    objective=item["objective"],
                    provenance="outer_train_contrast_vector",
                    contrast_family=item.get("contrast_family"),
                )
            for item in emb["x_features"]:
                _append_feature(
                    x_train_cols,
                    x_test_cols,
                    x_names,
                    feature_rows,
                    train=item["train"],
                    test=item["test"],
                    name=item["name"],
                    role="X",
                    source_family="embedding_contrast",
                    outer_fold=outer_fold,
                    objective=item["objective"],
                    provenance="outer_train_contrast_vector",
                    contrast_family=item.get("contrast_family"),
                )
            embedding_rows.extend(emb["metadata"])
            inner_model_rows.extend(emb.get("inner_model_rows", []))
            for item in [*emb["w_features"], *emb["x_features"]]:
                contrast_family = str(item.get("contrast_family") or "whole_cohort")
                prediction_frames.append(
                    _source_prediction_frame(
                        train_df,
                        test_df,
                        outer_fold=outer_fold,
                        source_name=(
                            f"embedding__{contrast_family}__{item['name']}"
                        ),
                        values={"score": (item["train"], item["test"])},
                    )
                )
            embedding_evidence = self._build_primary_embedding_contrast_evidence(
                discovery_df=train_df,
                y=y,
                t=t,
                pseudo_target=pseudo_target,
                t_resid=t_resid,
                importance=importance,
            )

        handoff_evidence = {
            "metrics": copy.deepcopy(metrics),
            "importance": importance,
            "embedding_contrast_evidence": embedding_evidence,
            "htr_evidence": htr_evidence,
        }

        return _FeatureBundle(
            x_train=_column_matrix(x_train_cols, len(train_df)),
            x_test=_column_matrix(x_test_cols, len(test_df)),
            w_train=_column_matrix(w_train_cols, len(train_df)),
            w_test=_column_matrix(w_test_cols, len(test_df)),
            x_names=x_names,
            w_names=w_names,
            feature_rows=feature_rows,
            prediction_frames=prediction_frames,
            embedding_rows=embedding_rows,
            metrics=metrics,
            handoff_evidence=handoff_evidence,
            inner_model_rows=inner_model_rows,
        )

    def _primary_bow_metrics(
        self,
        *,
        discovery_df: pd.DataFrame,
        y: np.ndarray,
        t: np.ndarray,
        e_hat: np.ndarray,
        m_hat: np.ndarray,
        pseudo_target: np.ndarray,
        tau_hat: np.ndarray,
        y_resid: np.ndarray,
        t_resid: np.ndarray,
    ) -> Dict[str, Any]:
        r_loss = (
            np.asarray(y_resid, dtype=float) - np.asarray(tau_hat, dtype=float) * t_resid
        ) ** 2
        r_loss_at_zero = np.asarray(y_resid, dtype=float) ** 2
        metrics: Dict[str, Any] = {
            "treatment_auroc": _safe_roc_auc(t, e_hat),
            "pseudo_target_mean": _finite_or_none(np.mean(pseudo_target)),
            "pseudo_target_std": _finite_or_none(np.std(pseudo_target)),
            "tau_hat_mean": _finite_or_none(np.mean(tau_hat)),
            "tau_hat_std": _finite_or_none(np.std(tau_hat)),
            "r_loss_mean": _finite_or_none(np.mean(r_loss)),
            "r_loss_at_zero_mean": _finite_or_none(np.mean(r_loss_at_zero)),
            "r_loss_improvement": _finite_or_none(np.mean(r_loss_at_zero) - np.mean(r_loss)),
            "pseudo_target_construction": (
                "Stage 1 ensemble nuisance predictions, then " "(Y - mean_m_hat) / (T - mean_e_hat)"
            ),
        }
        try:
            metrics["treatment_brier"] = _finite_or_none(brier_score_loss(t, e_hat))
        except Exception:
            pass
        try:
            metrics["treatment_log_loss"] = _finite_or_none(log_loss(t, e_hat))
        except Exception:
            pass
        if str(self.config.outcome_type).lower() == "continuous":
            metrics["outcome_rmse"] = _finite_or_none(np.sqrt(mean_squared_error(y, m_hat)))
        else:
            metrics["outcome_auroc"] = _safe_roc_auc(y, m_hat)
            try:
                metrics["outcome_brier"] = _finite_or_none(brier_score_loss(y, m_hat))
            except Exception:
                pass
        return metrics

    def _fit_primary_feature_importance_models(
        self,
        *,
        texts: Sequence[str],
        y: np.ndarray,
        t: np.ndarray,
        pseudo_target: np.ndarray,
        pseudo_target_sample_weight: Optional[np.ndarray],
        view: BoWViewConfig,
    ) -> Dict[str, Any]:
        vectorizer_params = _vectorizer_params(view)
        vectorizer = _make_bow_vectorizer(vectorizer_params)
        x_model = vectorizer.fit_transform(texts)
        features = np.asarray(vectorizer.get_feature_names_out())

        treatment_model = None
        treatment_constant = None
        if len(np.unique(np.asarray(t, dtype=int))) < 2:
            treatment_coef = np.zeros(len(features), dtype=float)
            treatment_constant = float(np.mean(np.asarray(t, dtype=float)))
            treatment_prediction = np.full(len(t), treatment_constant, dtype=float)
        else:
            treatment_model = _make_bow_classifier(_model_params(view), random_state=101)
            treatment_model.fit(x_model, np.asarray(t, dtype=int))
            treatment_coef = _model_feature_scores(treatment_model, len(features))
            treatment_prediction = treatment_model.predict_proba(x_model)[:, 1]

        outcome_model = None
        outcome_constant = None
        if str(self.config.outcome_type).lower() == "continuous":
            outcome_model = _make_bow_regressor(_model_params(view), random_state=202)
            outcome_model.fit(x_model, y)
            outcome_coef = _model_feature_scores(outcome_model, len(features))
            outcome_prediction = outcome_model.predict(x_model)
            outcome_classification = False
        elif len(np.unique(np.asarray(y, dtype=int))) < 2:
            outcome_coef = np.zeros(len(features), dtype=float)
            outcome_constant = float(np.mean(np.asarray(y, dtype=float)))
            outcome_prediction = np.full(len(y), outcome_constant, dtype=float)
            outcome_classification = True
        else:
            outcome_model = _make_bow_classifier(_model_params(view), random_state=202)
            outcome_model.fit(x_model, np.asarray(y, dtype=int))
            outcome_coef = _model_feature_scores(outcome_model, len(features))
            outcome_prediction = outcome_model.predict_proba(x_model)[:, 1]
            outcome_classification = True

        effect_model = _make_bow_regressor(_model_params(view), random_state=303)
        _fit_regressor(
            effect_model,
            x_model,
            pseudo_target,
            sample_weight=pseudo_target_sample_weight,
            unsupported_sample_weight_policy=(
                view.unsupported_sample_weight_policy
            ),
        )
        effect_coef = _model_feature_scores(effect_model, len(features))
        effect_prediction = effect_model.predict(x_model)

        if self.bow_native_capture_sink is not None:
            common = {
                "view_name": view.name,
                "view_config": _bow_view_to_dict(view),
                "vectorizer_params": vectorizer_params,
                "vectorizer": vectorizer,
            }
            self.bow_native_capture_sink.record_full_fit(
                **common,
                objective="treatment_importance",
                seed=101,
                target_values=t,
                sample_weight=None,
                learner=treatment_model,
                classification=True,
                constant_prediction=treatment_constant,
                fit_prediction=treatment_prediction,
            )
            self.bow_native_capture_sink.record_full_fit(
                **common,
                objective="outcome_importance",
                seed=202,
                target_values=y,
                sample_weight=None,
                learner=outcome_model,
                classification=outcome_classification,
                constant_prediction=outcome_constant,
                fit_prediction=outcome_prediction,
            )
            self.bow_native_capture_sink.record_full_fit(
                **common,
                objective="effect_weighted_r_importance",
                seed=303,
                target_values=pseudo_target,
                sample_weight=pseudo_target_sample_weight,
                learner=effect_model,
                classification=False,
                constant_prediction=None,
                fit_prediction=effect_prediction,
            )

        top_n = int(self.nn_config.top_n_features)
        confounder_score = np.abs(treatment_coef) * np.abs(outcome_coef)
        return {
            "view_name": str(view.name),
            "view_config": _bow_view_to_dict(view),
            "n_features": int(len(features)),
            "n_bow_features": int(len(features)),
            "n_prespecified_features": 0,
            "n_prespecified_raw_features": 0,
            "prespecified_raw_feature_names": [],
            "phrase_features": _top_phrase_feature_rows(
                features,
                top_n=top_n,
                treatment_coef=treatment_coef,
                outcome_coef=outcome_coef,
                pseudo_target_coef=effect_coef,
                confounder_score=confounder_score,
            ),
            "confounder_overlap": _top_feature_rows(
                features,
                confounder_score,
                top_n,
                treatment_coef=treatment_coef,
                outcome_coef=outcome_coef,
            ),
            "treatment_positive": _top_feature_rows(features, treatment_coef, top_n),
            "treatment_negative": _top_feature_rows(
                features,
                treatment_coef,
                top_n,
                descending=False,
            ),
            "outcome_positive": _top_feature_rows(features, outcome_coef, top_n),
            "outcome_negative": _top_feature_rows(
                features,
                outcome_coef,
                top_n,
                descending=False,
            ),
            "pseudo_target_positive": _top_feature_rows(features, effect_coef, top_n),
            "pseudo_target_negative": _top_feature_rows(
                features,
                effect_coef,
                top_n,
                descending=False,
            ),
        }

    def _build_primary_embedding_contrast_evidence(
        self,
        *,
        discovery_df: pd.DataFrame,
        y: np.ndarray,
        t: np.ndarray,
        pseudo_target: np.ndarray,
        t_resid: np.ndarray,
        importance: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not self._embedding_contrast_enabled():
            return {}
        generator = self._embedding_generator()
        generator.prepare(self.dataset)
        ordered_fit_rows = tuple(
            discovery_df["_oci_row_id"].astype(int).tolist()
        )
        generator.bind_cluster_fit_context(
            ordered_fit_row_ids=ordered_fit_rows,
            canonical_group_seed=derive_discovery_seed(
                int(self.config.seed),
                ordered_fit_rows,
            ),
        )
        return generator.build_evidence(
            discovery_df=discovery_df,
            y=y,
            t=t,
            pseudo_target=[pseudo_target],
            t_resid=[t_resid],
            pseudo_target_names=["stage1_ensemble_mean_nuisance"],
            importance=importance,
        )

    def _build_primary_agent_context(
        self,
        *,
        outer_fold: int,
        discovery_df: pd.DataFrame,
        metrics: Dict[str, Any],
        importance: Dict[str, Any],
        embedding_evidence: Optional[Dict[str, Any]] = None,
        htr_evidence: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        agent_context_mode = (
            str(getattr(self.nn_config, "agent_context_mode", "evidence_digest") or "")
            .strip()
            .lower()
        )
        if agent_context_mode == "evidence_digest":
            context = _build_evidence_digest_agent_context(
                outer_fold=outer_fold,
                feature_discovery_methods=self._enabled_feature_discovery_methods(),
                max_proposals=int(self.nn_config.candidate_proposals_per_fold),
                clinical_question=self.config.clinical_question,
                treatment_column=self.config.treatment_column,
                outcome_column=self.config.outcome_column,
                outcome_type=self.config.outcome_type,
                current_features=[],
                metrics=metrics,
                importance=importance,
                clinical_text_examples=_clinical_text_examples(
                    discovery_df,
                    self.config.text_column,
                    n_examples=self.search_config.clinical_text_examples_per_prompt,
                    max_chars=self.search_config.clinical_text_example_chars,
                ),
                embedding_evidence=embedding_evidence,
                htr_evidence=htr_evidence,
                handoff_provenance={
                    "source": "multi_model_forest_stage1_primary_text_models",
                    "raw_text_modeling_reused_for_agentic_stage": True,
                },
            )
            prompt_chars = len(json.dumps(context, separators=(",", ":"), default=str))
            logger.info(
                "Multi-model forest primary evidence-digest context outer_fold=%s: %.1fK JSON chars",
                outer_fold,
                prompt_chars / 1000.0,
            )
            return context
        instructions = [
            "You are generating candidate variables from empirical text evidence.",
            "The evidence was produced during Stage 1 primary text-model forest training.",
            "Suggest explicit pre-treatment patient-level variables, not raw text tokens.",
            "Use variables predictive of both treatment and outcome as confounders.",
            "Use variables predictive of the pseudo-target or R-stage signal as effect modifiers.",
            "Do not invent broad clinical inventory variables unsupported by the enabled evidence.",
            "Avoid near-duplicate aliases for the same extraction target.",
        ]
        if self._bow_enabled():
            instructions.append(
                "Review sparse bag-of-words feature importance across the Stage 1 views."
            )
        if self._embedding_contrast_enabled():
            instructions.append(
                "Use embedding_contrast_evidence as retrieved chunk evidence, not as a direct vector interpretation."
            )
        if self._htr_enabled():
            instructions.append(
                "Use htr_attention_evidence from the Stage 1 HTR nuisance and R-stage models as neural text evidence."
            )
        if self._matched_pair_uplift_enabled():
            instructions.append(
                "Use matched_pair_uplift evidence as direct effect-modifier evidence: "
                "BoW uplift coefficients and HTR pair-uplift attention are trained on "
                "treated/control matched pairs and target treated-patient outcome under treatment."
            )
        context: Dict[str, Any] = {
            "prompt_version": "multi_model_agentic_forest_v1",
            "outer_fold": int(outer_fold),
            "feature_discovery_methods": self._enabled_feature_discovery_methods(),
            "max_proposals": int(self.nn_config.candidate_proposals_per_fold),
            "clinical_question": self.config.clinical_question,
            "estimand": {
                "treatment_column": self.config.treatment_column,
                "outcome_column": self.config.outcome_column,
                "outcome_type": self.config.outcome_type,
            },
            "instructions": instructions,
            "current_features": [],
            "model_diagnostics": _agent_visible_metrics(metrics),
            "feature_importance": importance,
            "clinical_text_examples": _clinical_text_examples(
                discovery_df,
                self.config.text_column,
                n_examples=self.search_config.clinical_text_examples_per_prompt,
                max_chars=self.search_config.clinical_text_example_chars,
            ),
            "response_contract": {
                "proposals": [
                    {
                        "action": "add",
                        "name": "snake_case_variable_name",
                        "type": "categorical|continuous",
                        "categories": ["category_a", "category_b"],
                        "roles": ["confounder", "effect_modifier"],
                        "description": "exact pre-treatment extraction target",
                        "rationale": "which enabled evidence supports this variable",
                        "expected_signal": "treatment, outcome, or pseudo-target signal expected",
                    }
                ]
            },
            "handoff_provenance": {
                "source": "multi_model_forest_stage1_primary_text_models",
                "raw_text_modeling_reused_for_agentic_stage": True,
            },
        }
        if embedding_evidence:
            context["embedding_contrast_evidence"] = embedding_evidence
        if htr_evidence:
            context["htr_attention_evidence"] = htr_evidence
        compact_context = _compact_multi_model_agent_context(context)
        prompt_chars = len(json.dumps(compact_context, separators=(",", ":"), default=str))
        logger.info(
            "Multi-model forest primary handoff context outer_fold=%s: %.1fK JSON chars",
            outer_fold,
            prompt_chars / 1000.0,
        )
        return compact_context

    def _fit_bow_binary_train_test(
        self,
        texts_train: Sequence[str],
        texts_test: Sequence[str],
        labels: np.ndarray,
        *,
        outer_fold: int,
        view: BoWViewConfig,
        view_index: int,
        label_name: str,
        train_row_ids: Optional[Sequence[Any]] = None,
        test_row_ids: Optional[Sequence[Any]] = None,
        checkpoint_namespace: Optional[str] = None,
        checkpoint_dependencies: Sequence[str] = (),
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        labels = np.asarray(labels, dtype=int)
        oof = np.full(len(labels), np.nan, dtype=float)
        split_items = list(
            enumerate(
                _binary_split_items(
                    labels,
                    requested_folds=int(self.nn_config.nuisance_folds),
                    random_state=11_000
                    + 100 * int(outer_fold)
                    + 1_000 * int(view_index)
                    + (1 if label_name == "outcome" else 2),
                ),
                start=1,
            )
        )
        vectorizer_params = _vectorizer_params(view)
        model_params = _model_params(view)
        e_clip = float(self.nn_config.e_clip)
        capture_sink = self.bow_native_capture_sink
        n_jobs = self._fold_n_jobs(len(split_items))
        logger.info(
            "Outer fold %s BoW binary %s view=%s model=%s folds=%s n_jobs=%s " "backend=%s",
            outer_fold,
            label_name,
            view.name,
            view.bow_model,
            len(split_items),
            n_jobs,
            self._parallel_backend_name(),
        )

        def compute_fold(fold: int, fit_pos: np.ndarray, heldout_pos: np.ndarray):
            fit_pos = np.asarray(fit_pos, dtype=int)
            heldout_pos = np.asarray(heldout_pos, dtype=int)
            vectorizer = None
            model = None
            constant_prediction = None
            if len(np.unique(labels[fit_pos])) < 2:
                constant_prediction = float(np.mean(labels[fit_pos]))
                heldout_pred = np.full(
                    len(heldout_pos),
                    constant_prediction,
                    dtype=float,
                )
                test_pred = np.full(len(texts_test), constant_prediction, dtype=float)
            else:
                vectorizer = _make_bow_vectorizer(vectorizer_params)
                fit_texts = [texts_train[int(pos)] for pos in fit_pos]
                heldout_texts = [texts_train[int(pos)] for pos in heldout_pos]
                x_fit = vectorizer.fit_transform(fit_texts)
                x_heldout = vectorizer.transform(heldout_texts)
                x_test = vectorizer.transform(texts_test)
                model = _make_bow_classifier(model_params, random_state=17 + int(fold))
                model.fit(x_fit, labels[fit_pos])
                heldout_pred = model.predict_proba(x_heldout)[:, 1]
                test_pred = model.predict_proba(x_test)[:, 1]
            heldout_pred = np.clip(
                heldout_pred,
                e_clip,
                1.0 - e_clip,
            )
            test_pred = np.clip(
                test_pred,
                e_clip,
                1.0 - e_clip,
            )
            if capture_sink is not None:
                capture_sink.record_fold(
                    family="bow_nuisance",
                    objective=f"{label_name}_nuisance",
                    view_name=view.name,
                    view_config=_bow_view_to_dict(view),
                    fold=int(fold),
                    fit_positions=fit_pos,
                    validation_positions=heldout_pos,
                    seed=17 + int(fold),
                    target_values=labels,
                    sample_weight=None,
                    vectorizer_params=vectorizer_params,
                    vectorizer=vectorizer,
                    learner=model,
                    classification=True,
                    constant_prediction=constant_prediction,
                    validation_prediction=heldout_pred,
                    heldout_prediction=test_pred,
                )
            return {
                "fold": int(fold),
                "heldout_pos": heldout_pos,
                "heldout_pred": heldout_pred,
                "test_pred": test_pred,
                "evidence": {
                    "outer_fold": int(outer_fold),
                    "inner_fold": int(fold),
                    "source_family": "bow",
                    "view_name": view.name,
                    "view_config": _bow_view_to_dict(view),
                    "objective": f"{label_name}_nuisance",
                    "target_name": label_name,
                    "model": str(view.bow_model),
                    "train_rows": int(len(fit_pos)),
                    "heldout_rows": int(len(heldout_pos)),
                    "outer_test_rows": int(len(texts_test)),
                    "heldout_auroc": _safe_roc_auc(labels[heldout_pos], heldout_pred),
                    "prediction_provenance": "inner_fold_model_heldout_and_outer_test",
                },
            }

        def run_fold(fold: int, fit_pos: np.ndarray, heldout_pos: np.ndarray):
            store = self._active_checkpoint_store()
            if store is None or checkpoint_namespace is None:
                return compute_fold(fold, fit_pos, heldout_pos)
            fit_pos = np.asarray(fit_pos, dtype=int)
            heldout_pos = np.asarray(heldout_pos, dtype=int)
            fit_ids = np.asarray(
                np.arange(len(labels)) if train_row_ids is None else train_row_ids
            )
            heldout_ids = np.asarray(
                np.arange(len(texts_test)) if test_row_ids is None else test_row_ids
            )
            if len(fit_ids) != len(labels) or len(heldout_ids) != len(texts_test):
                raise ValueError("BoW binary checkpoint row ID counts do not match inputs")
            return store.run(
                f"{checkpoint_namespace}/fold_{int(fold):03d}",
                lambda: compute_fold(fold, fit_pos, heldout_pos),
                parameters={
                    "outer_fold": int(outer_fold),
                    "inner_fold": int(fold),
                    "total_folds": int(len(split_items)),
                    "view_index": int(view_index),
                    "view": _bow_view_to_dict(view),
                    "label_name": label_name,
                },
                dependencies=tuple(checkpoint_dependencies),
                rows={
                    "fit": row_id_identity(fit_ids[fit_pos]),
                    "validation": row_id_identity(fit_ids[heldout_pos]),
                    "heldout": row_id_identity(heldout_ids),
                },
                validator=lambda payload: _validate_prediction_fold_payload(
                    payload,
                    len(heldout_pos),
                    len(texts_test),
                    "BoW binary fold",
                ),
            ).value

        fold_results = self._run_fold_tasks(run_fold, split_items)
        test_predictions = []
        evidence_rows = []
        for result in fold_results:
            heldout_pos = result["heldout_pos"]
            oof[heldout_pos] = result["heldout_pred"]
            test_predictions.append(np.asarray(result["test_pred"], dtype=float))
            evidence_rows.append(result["evidence"])
        test_pred = np.nanmean(np.vstack(test_predictions), axis=0)
        return (
            np.clip(oof, self.nn_config.e_clip, 1.0 - self.nn_config.e_clip),
            np.clip(test_pred, self.nn_config.e_clip, 1.0 - self.nn_config.e_clip),
            evidence_rows,
        )

    def _fit_bow_regression_train_test(
        self,
        texts_train: Sequence[str],
        texts_test: Sequence[str],
        values: np.ndarray,
        sample_weight: Optional[np.ndarray],
        *,
        outer_fold: int,
        view: BoWViewConfig,
        view_index: int,
        target_name: str,
        seed_offset: int = 0,
        train_row_ids: Optional[Sequence[Any]] = None,
        test_row_ids: Optional[Sequence[Any]] = None,
        checkpoint_namespace: Optional[str] = None,
        checkpoint_dependencies: Sequence[str] = (),
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        values = np.asarray(values, dtype=float)
        oof = np.full(len(values), np.nan, dtype=float)
        folds = _bounded_fold_count(
            int(
                self.nn_config.effect_folds
                if "effect" in target_name
                else self.nn_config.nuisance_folds
            ),
            len(values),
        )
        splitter = KFold(
            n_splits=folds,
            shuffle=True,
            random_state=13_000
            + int(seed_offset)
            + 100 * int(outer_fold)
            + 1_000 * int(view_index),
        )
        split_items = list(enumerate(splitter.split(texts_train), start=1))
        vectorizer_params = _vectorizer_params(view)
        model_params = _model_params(view)
        capture_sink = self.bow_native_capture_sink
        n_jobs = self._fold_n_jobs(len(split_items))
        logger.info(
            "Outer fold %s BoW regression %s view=%s model=%s folds=%s n_jobs=%s " "backend=%s",
            outer_fold,
            target_name,
            view.name,
            view.bow_model,
            len(split_items),
            n_jobs,
            self._parallel_backend_name(),
        )

        def compute_fold(fold: int, fit_pos: np.ndarray, heldout_pos: np.ndarray):
            fit_pos = np.asarray(fit_pos, dtype=int)
            heldout_pos = np.asarray(heldout_pos, dtype=int)
            vectorizer = _make_bow_vectorizer(vectorizer_params)
            fit_texts = [texts_train[int(pos)] for pos in fit_pos]
            heldout_texts = [texts_train[int(pos)] for pos in heldout_pos]
            x_fit = vectorizer.fit_transform(fit_texts)
            x_heldout = vectorizer.transform(heldout_texts)
            x_test = vectorizer.transform(texts_test)
            model = _make_bow_regressor(
                model_params,
                random_state=17 + int(seed_offset) + int(fold),
            )
            fold_weight = None
            if sample_weight is not None:
                fold_weight = np.asarray(sample_weight, dtype=float)[fit_pos]
            _fit_regressor(
                model,
                x_fit,
                values[fit_pos],
                sample_weight=fold_weight,
                unsupported_sample_weight_policy=str(
                    model_params["unsupported_sample_weight_policy"]
                ),
            )
            heldout_pred = model.predict(x_heldout)
            test_pred = model.predict(x_test)
            heldout_values = values[heldout_pos]
            if capture_sink is not None:
                capture_sink.record_fold(
                    family=("bow_r_loss" if "effect" in target_name else "bow_nuisance"),
                    objective=("outcome_nuisance" if target_name == "outcome" else target_name),
                    view_name=view.name,
                    view_config=_bow_view_to_dict(view),
                    fold=int(fold),
                    fit_positions=fit_pos,
                    validation_positions=heldout_pos,
                    seed=17 + int(seed_offset) + int(fold),
                    target_values=values,
                    sample_weight=sample_weight,
                    vectorizer_params=vectorizer_params,
                    vectorizer=vectorizer,
                    learner=model,
                    classification=False,
                    constant_prediction=None,
                    validation_prediction=heldout_pred,
                    heldout_prediction=test_pred,
                )
            return {
                "fold": int(fold),
                "heldout_pos": heldout_pos,
                "heldout_pred": heldout_pred,
                "test_pred": test_pred,
                "evidence": {
                    "outer_fold": int(outer_fold),
                    "inner_fold": int(fold),
                    "source_family": "bow",
                    "view_name": view.name,
                    "view_config": _bow_view_to_dict(view),
                    "objective": target_name,
                    "target_name": target_name,
                    "model": str(view.bow_model),
                    "train_rows": int(len(fit_pos)),
                    "heldout_rows": int(len(heldout_pos)),
                    "outer_test_rows": int(len(texts_test)),
                    "heldout_rmse": _finite_or_none(
                        np.sqrt(mean_squared_error(heldout_values, heldout_pred))
                    ),
                    "prediction_provenance": "inner_fold_model_heldout_and_outer_test",
                },
            }

        def run_fold(fold: int, fit_pos: np.ndarray, heldout_pos: np.ndarray):
            store = self._active_checkpoint_store()
            if store is None or checkpoint_namespace is None:
                return compute_fold(fold, fit_pos, heldout_pos)
            fit_pos = np.asarray(fit_pos, dtype=int)
            heldout_pos = np.asarray(heldout_pos, dtype=int)
            fit_ids = np.asarray(
                np.arange(len(values)) if train_row_ids is None else train_row_ids
            )
            heldout_ids = np.asarray(
                np.arange(len(texts_test)) if test_row_ids is None else test_row_ids
            )
            if len(fit_ids) != len(values) or len(heldout_ids) != len(texts_test):
                raise ValueError("BoW regression checkpoint row ID counts do not match inputs")
            return store.run(
                f"{checkpoint_namespace}/fold_{int(fold):03d}",
                lambda: compute_fold(fold, fit_pos, heldout_pos),
                parameters={
                    "outer_fold": int(outer_fold),
                    "inner_fold": int(fold),
                    "total_folds": int(len(split_items)),
                    "view_index": int(view_index),
                    "view": _bow_view_to_dict(view),
                    "target_name": target_name,
                    "seed_offset": int(seed_offset),
                    "sample_weighted": sample_weight is not None,
                },
                dependencies=tuple(checkpoint_dependencies),
                rows={
                    "fit": row_id_identity(fit_ids[fit_pos]),
                    "validation": row_id_identity(fit_ids[heldout_pos]),
                    "heldout": row_id_identity(heldout_ids),
                },
                validator=lambda payload: _validate_prediction_fold_payload(
                    payload,
                    len(heldout_pos),
                    len(texts_test),
                    "BoW regression fold",
                ),
            ).value

        fold_results = self._run_fold_tasks(run_fold, split_items)
        test_predictions = []
        evidence_rows = []
        for result in fold_results:
            heldout_pos = result["heldout_pos"]
            oof[heldout_pos] = result["heldout_pred"]
            test_predictions.append(np.asarray(result["test_pred"], dtype=float))
            evidence_rows.append(result["evidence"])
        test_pred = np.nanmean(np.vstack(test_predictions), axis=0)
        return oof, test_pred, evidence_rows

    def _embedding_feature_bundle(
        self,
        *,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        y: np.ndarray,
        t: np.ndarray,
        pseudo_target: np.ndarray,
        t_resid: np.ndarray,
        outer_fold: int,
    ) -> Dict[str, Any]:
        generator = self._embedding_generator()
        generator.prepare(self.dataset)
        train_positions = generator._positions_for_frame(train_df)
        test_positions = generator._positions_for_frame(test_df)
        train_patient = generator._patient_embeddings(train_positions)
        train_patient = _residualize_embeddings(
            train_patient,
            train_df,
            self.nn_config.embedding_contrast.residualize_columns,
        )
        train_patient = _normalize_rows(train_patient)
        folds = _bounded_fold_count(int(self.nn_config.nuisance_folds), len(train_df))
        splitter = KFold(n_splits=folds, shuffle=True, random_state=40_000 + int(outer_fold))
        feature_map: Dict[str, Dict[str, Any]] = {}
        metadata: List[Dict[str, Any]] = []
        inner_model_rows: List[Dict[str, Any]] = []

        for inner_fold, (fit_pos, heldout_pos) in enumerate(splitter.split(train_df), start=1):
            fit_pos = np.asarray(fit_pos, dtype=int)
            heldout_pos = np.asarray(heldout_pos, dtype=int)
            directions, fold_metadata = self._embedding_directions(
                patient_embeddings=train_patient[fit_pos],
                fit_row_ids=tuple(
                    map(
                        int,
                        train_df["_oci_row_id"].to_numpy()[fit_pos],
                    )
                ),
                y=np.asarray(y, dtype=float)[fit_pos],
                t=np.asarray(t, dtype=float)[fit_pos],
                pseudo_target=np.asarray(pseudo_target, dtype=float)[fit_pos],
                t_resid=np.asarray(t_resid, dtype=float)[fit_pos],
                outer_fold=1000 * int(outer_fold) + int(inner_fold),
            )
            for row in fold_metadata:
                row = dict(row)
                row["outer_fold"] = int(outer_fold)
                row["inner_fold"] = int(inner_fold)
                row["prediction_provenance"] = "inner_fold_contrast_direction"
                metadata.append(row)
            inner_model_rows.append(
                {
                    "outer_fold": int(outer_fold),
                    "inner_fold": int(inner_fold),
                    "source_family": "embedding_contrast",
                    "objective": "contrast_direction",
                    "target_name": "embedding_contrast",
                    "train_rows": int(len(fit_pos)),
                    "heldout_rows": int(len(heldout_pos)),
                    "outer_test_rows": int(len(test_df)),
                    "n_contrast_directions": int(len(directions)),
                    "prediction_provenance": "inner_fold_model_heldout_and_outer_test",
                }
            )
            heldout_positions = [int(train_positions[int(pos)]) for pos in heldout_pos]
            for direction in directions:
                heldout_mean, heldout_max = self._chunk_similarity_features(
                    generator,
                    heldout_positions,
                    direction["direction"],
                )
                test_mean, test_max = self._chunk_similarity_features(
                    generator,
                    test_positions,
                    direction["direction"],
                )
                base_name = f"embedding__{direction['name']}"
                for stat, heldout_values, test_values in [
                    ("mean_cosine", heldout_mean, test_mean),
                    ("max_cosine", heldout_max, test_max),
                ]:
                    feature_name = f"{base_name}__{stat}"
                    entry = feature_map.setdefault(
                        feature_name,
                        {
                            "name": feature_name,
                            "role": direction["role"],
                            "objective": direction["objective"],
                            "contrast_family": direction["contrast_family"],
                            "train": np.full(len(train_df), np.nan, dtype=np.float32),
                            "test_predictions": [],
                        },
                    )
                    entry["train"][heldout_pos] = heldout_values
                    entry["test_predictions"].append(np.asarray(test_values, dtype=np.float32))

        w_features = []
        x_features = []
        for entry in feature_map.values():
            train_values = np.asarray(entry["train"], dtype=np.float32)
            train_values = np.where(np.isfinite(train_values), train_values, 0.0)
            test_values = (
                np.nanmean(np.vstack(entry["test_predictions"]), axis=0)
                if entry["test_predictions"]
                else np.zeros(len(test_df), dtype=np.float32)
            )
            target = w_features if entry["role"] == "W" else x_features
            target.append(
                {
                    "name": entry["name"],
                    "train": train_values,
                    "test": np.asarray(test_values, dtype=np.float32),
                    "objective": entry["objective"],
                    "contrast_family": entry["contrast_family"],
                }
            )
        return {
            "w_features": w_features,
            "x_features": x_features,
            "metadata": metadata,
            "inner_model_rows": inner_model_rows,
        }

    def _embedding_directions(
        self,
        *,
        patient_embeddings: np.ndarray,
        fit_row_ids: Sequence[int],
        y: np.ndarray,
        t: np.ndarray,
        pseudo_target: np.ndarray,
        t_resid: np.ndarray,
        outer_fold: int,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        directions: List[Dict[str, Any]] = []
        metadata: List[Dict[str, Any]] = []
        ordered_fit_rows = tuple(map(int, fit_row_ids))
        if (
            len(ordered_fit_rows) != len(patient_embeddings)
            or len(set(ordered_fit_rows)) != len(ordered_fit_rows)
        ):
            raise ValueError(
                "embedding direction fit rows must be exact, ordered, and unique"
            )
        finite = np.all(np.isfinite(patient_embeddings), axis=1)
        treatment_labels, treatment_mask = _binary_labels(t)
        t_direction, t_counts = _binary_mean_difference_direction(
            patient_embeddings,
            treatment_labels,
            treatment_mask & finite,
        )
        outcome_labels, outcome_mask = (
            _tail_labels(y, float(self.nn_config.embedding_contrast.pseudo_target_quantile))
            if str(self.config.outcome_type).lower() == "continuous"
            else _binary_labels(y)
        )
        y_direction, y_counts = _binary_mean_difference_direction(
            patient_embeddings,
            outcome_labels,
            outcome_mask & finite,
        )
        if t_direction is not None:
            self._add_embedding_direction(
                directions,
                metadata,
                outer_fold,
                name="global_treatment_contrast",
                direction=t_direction,
                role="W",
                objective="treatment_confounder",
                contrast_family="global_marginal_treatment",
                counts=t_counts,
            )
        if y_direction is not None:
            self._add_embedding_direction(
                directions,
                metadata,
                outer_fold,
                name="global_outcome_contrast",
                direction=y_direction,
                role="W",
                objective="outcome_confounder",
                contrast_family="global_marginal_outcome",
                counts=y_counts,
            )
        if t_direction is not None and y_direction is not None:
            confounder = 0.5 * _normalize_vector(t_direction) + 0.5 * _normalize_vector(y_direction)
            self._add_embedding_direction(
                directions,
                metadata,
                outer_fold,
                name="global_confounder_average",
                direction=confounder,
                role="W",
                objective="treatment_outcome_confounder_average",
                contrast_family="global_marginal_confounder_average",
                counts={"treatment": t_counts, "outcome": y_counts},
            )

        pseudo_labels, pseudo_mask = _tail_labels(
            pseudo_target,
            float(self.nn_config.embedding_contrast.pseudo_target_quantile),
        )
        pseudo_weights = np.square(np.asarray(t_resid, dtype=float))
        pseudo_direction, pseudo_counts = _weighted_binary_direction(
            patient_embeddings,
            pseudo_labels,
            pseudo_mask & finite,
            (
                pseudo_weights
                if bool(self.nn_config.embedding_contrast.pseudo_target_weighted)
                else None
            ),
        )
        if pseudo_direction is not None:
            self._add_embedding_direction(
                directions,
                metadata,
                outer_fold,
                name="global_r_pseudo_target_contrast",
                direction=pseudo_direction,
                role="X",
                objective="r_pseudo_outcome",
                contrast_family="global_r_pseudo_target",
                counts=pseudo_counts,
            )
        orthogonal_score = np.asarray(pseudo_target, dtype=float) * np.square(
            np.asarray(t_resid, dtype=float)
        )
        score_labels, score_mask = _tail_labels(
            orthogonal_score,
            float(self.nn_config.embedding_contrast.pseudo_target_quantile),
        )
        score_direction, score_counts = _binary_mean_difference_direction(
            patient_embeddings,
            score_labels,
            score_mask & finite,
        )
        if score_direction is not None:
            self._add_embedding_direction(
                directions,
                metadata,
                outer_fold,
                name="global_orthogonal_r_score_contrast",
                direction=score_direction,
                role="X",
                objective="orthogonal_r_score",
                contrast_family="global_orthogonal_r_score",
                counts=score_counts,
            )

        residual_interaction = self._residualized_interaction_direction(
            patient_embeddings,
            y,
            t,
            treatment_labels,
            treatment_mask,
            outcome_labels,
            outcome_mask,
            t_direction,
            y_direction,
            finite,
        )
        if residual_interaction is not None:
            self._add_embedding_direction(
                directions,
                metadata,
                outer_fold,
                name="global_residualized_treatment_outcome_interaction",
                direction=residual_interaction,
                role="X",
                objective="residualized_treatment_outcome_interaction",
                contrast_family="global_residualized_interaction",
                counts={},
            )

        if bool(self.nn_config.embedding_contrast.include_cluster_contrast_vectors):
            directions.extend(
                self._cluster_embedding_directions(
                    patient_embeddings=patient_embeddings,
                    y=y,
                    t=t,
                    outcome_labels=outcome_labels,
                    outcome_mask=outcome_mask,
                    treatment_labels=treatment_labels,
                    treatment_mask=treatment_mask,
                    finite=finite,
                    metadata=metadata,
                    outer_fold=outer_fold,
                    canonical_group_seed=derive_discovery_seed(
                        int(self.config.seed),
                        ordered_fit_rows,
                    ),
                )
            )
        return directions, metadata

    def _add_embedding_direction(
        self,
        directions: List[Dict[str, Any]],
        metadata: List[Dict[str, Any]],
        outer_fold: int,
        *,
        name: str,
        direction: np.ndarray,
        role: str,
        objective: str,
        contrast_family: str,
        counts: Any,
    ) -> None:
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(norm) or norm <= 0.0:
            return
        direction = _normalize_vector(direction)
        directions.append(
            {
                "name": name,
                "direction": direction,
                "role": role,
                "objective": objective,
                "contrast_family": contrast_family,
            }
        )
        metadata.append(
            {
                "outer_fold": int(outer_fold),
                "name": name,
                "role": role,
                "objective": objective,
                "contrast_family": contrast_family,
                "direction_norm": float(np.linalg.norm(direction)),
                "counts": counts,
            }
        )

    def _cluster_embedding_directions(
        self,
        *,
        patient_embeddings: np.ndarray,
        y: np.ndarray,
        t: np.ndarray,
        outcome_labels: np.ndarray,
        outcome_mask: np.ndarray,
        treatment_labels: np.ndarray,
        treatment_mask: np.ndarray,
        finite: np.ndarray,
        metadata: List[Dict[str, Any]],
        outer_fold: int,
        canonical_group_seed: int,
    ) -> List[Dict[str, Any]]:
        cfg = self.nn_config.embedding_contrast
        scientific = _cluster_local_scientific_config(cfg)
        dtype = np.dtype(scientific.computation_dtype)
        n_usable = int(np.sum(finite))
        n_clusters = int(scientific.requested_cluster_count)
        if n_usable < n_clusters * int(scientific.minimum_cluster_size):
            raise ValueError(
                "cluster-local Stage 1 feature fit cannot satisfy the exact "
                "configured cluster count and minimum support"
            )
        kmeans_parameters = _embedding_cluster_kmeans_parameters(
            cfg,
            n_usable=n_usable,
            canonical_group_seed=int(canonical_group_seed),
            n_clusters=n_clusters,
        )
        kmeans = MiniBatchKMeans(**kmeans_parameters)
        labels = np.full(len(patient_embeddings), -1, dtype=int)
        labels[finite] = kmeans.fit_predict(
            np.asarray(patient_embeddings[finite], dtype=dtype)
        )
        counts = np.bincount(labels[finite], minlength=n_clusters)
        treatment_items = []
        interaction_items = []
        for cluster_id in range(n_clusters):
            if int(counts[cluster_id]) < int(scientific.minimum_cluster_size):
                continue
            cluster_mask = labels == cluster_id
            local_mask = cluster_mask & treatment_mask & finite
            pos = local_mask & (treatment_labels == 1)
            neg = local_mask & (treatment_labels == 0)
            if int(np.sum(pos)) >= int(scientific.minimum_group_size) and int(
                np.sum(neg)
            ) >= int(scientific.minimum_group_size):
                direction = np.mean(patient_embeddings[pos], axis=0) - np.mean(
                    patient_embeddings[neg],
                    axis=0,
                )
                if float(np.linalg.norm(direction)) > 0:
                    treatment_items.append(
                        {
                            "cluster_id": int(cluster_id),
                            "n_cluster": int(counts[cluster_id]),
                            "direction": direction,
                        }
                    )
            interaction = self._cluster_local_interaction_direction(
                patient_embeddings,
                treatment_labels,
                treatment_mask,
                outcome_labels,
                outcome_mask,
                cluster_mask & finite,
            )
            if interaction is not None:
                interaction_items.append(
                    {
                        "cluster_id": int(cluster_id),
                        "n_cluster": int(counts[cluster_id]),
                        "direction": interaction,
                    }
                )
        result: List[Dict[str, Any]] = []
        minimum_local = int(
            scientific.minimum_distinct_local_clusters_per_family
        )
        if (
            len(treatment_items) < minimum_local
            or len(interaction_items) < minimum_local
        ):
            raise ValueError(
                "cluster-local Stage 1 feature fit lacks the configured "
                "independently supported local contrasts"
            )
        metadata.append(
            {
                "outer_fold": int(outer_fold),
                "name": "cluster_contrast_vectors",
                "n_usable_patients": n_usable,
                "cluster_counts": [int(value) for value in counts],
                "canonical_group_seed": int(canonical_group_seed),
                "kmeans_parameters": copy.deepcopy(kmeans_parameters),
                "scientific_configuration": scientific.as_dict(),
            }
        )
        result.extend(
            self._svd_cluster_components(
                items=treatment_items,
                role="W",
                objective="cluster_treatment_confounder",
                contrast_family="cluster_local_treatment_contrast_basis",
                prefix="cluster_confounder_treatment",
                metadata=metadata,
                outer_fold=outer_fold,
            )
        )
        result.extend(
            self._svd_cluster_components(
                items=interaction_items,
                role="X",
                objective="cluster_residualized_treatment_outcome_interaction",
                contrast_family="cluster_local_residualized_interaction_contrast_basis",
                prefix="cluster_effect_residualized_interaction",
                metadata=metadata,
                outer_fold=outer_fold,
            )
        )
        return result

    def _cluster_local_interaction_direction(
        self,
        patient_embeddings: np.ndarray,
        treatment_labels: np.ndarray,
        treatment_mask: np.ndarray,
        outcome_labels: np.ndarray,
        outcome_mask: np.ndarray,
        cluster_mask: np.ndarray,
    ) -> Optional[np.ndarray]:
        base = cluster_mask & treatment_mask & outcome_mask
        treated_positive = base & (treatment_labels == 1) & (outcome_labels == 1)
        treated_negative = base & (treatment_labels == 1) & (outcome_labels == 0)
        untreated_positive = base & (treatment_labels == 0) & (outcome_labels == 1)
        untreated_negative = base & (treatment_labels == 0) & (outcome_labels == 0)
        min_cell = int(
            _cluster_local_scientific_config(
                self.nn_config.embedding_contrast
            ).minimum_cell_size
        )
        if (
            min(
                int(np.sum(treated_positive)),
                int(np.sum(treated_negative)),
                int(np.sum(untreated_positive)),
                int(np.sum(untreated_negative)),
            )
            < min_cell
        ):
            return None
        raw = (
            np.mean(patient_embeddings[treated_positive], axis=0)
            - np.mean(patient_embeddings[treated_negative], axis=0)
            - np.mean(patient_embeddings[untreated_positive], axis=0)
            + np.mean(patient_embeddings[untreated_negative], axis=0)
        )
        t_dir, _ = _binary_mean_difference_direction(
            patient_embeddings,
            treatment_labels,
            cluster_mask & treatment_mask,
        )
        y_dir, _ = _binary_mean_difference_direction(
            patient_embeddings,
            outcome_labels,
            cluster_mask & outcome_mask,
        )
        if t_dir is None or y_dir is None:
            return None
        residual = _residualize_vector_from_basis(raw, [t_dir, y_dir])
        if float(np.linalg.norm(residual)) <= 0.0:
            return None
        return residual

    def _svd_cluster_components(
        self,
        *,
        items: Sequence[Dict[str, Any]],
        role: str,
        objective: str,
        contrast_family: str,
        prefix: str,
        metadata: List[Dict[str, Any]],
        outer_fold: int,
    ) -> List[Dict[str, Any]]:
        scientific = _cluster_local_scientific_config(
            self.nn_config.embedding_contrast
        )
        if len(items) < int(
            scientific.minimum_distinct_local_clusters_per_family
        ):
            raise ValueError(
                f"cluster-local {contrast_family} lacks configured support"
            )
        dtype = np.dtype(scientific.computation_dtype)
        matrix = np.vstack(
            [
                _normalize_rows_configured(
                    np.asarray(item["direction"], dtype=dtype).reshape(1, -1),
                    normalize=True,
                    epsilon=float(scientific.normalization_epsilon),
                    zero_vector_policy=scientific.zero_vector_policy,
                    dtype=scientific.computation_dtype,
                )[0]
                * np.sqrt(float(item["n_cluster"]))
                for item in items
            ]
        ).astype(dtype, copy=False)
        svd_parameters = {
            "full_matrices": bool(scientific.svd_full_matrices),
            "compute_uv": bool(scientific.svd_compute_uv),
            "hermitian": bool(scientific.svd_hermitian),
        }
        _left, singular_values, components = np.linalg.svd(
            matrix,
            **svd_parameters,
        )
        components = _canonicalize_svd_component_signs(
            components,
            policy=scientific.svd_sign_canonicalization_policy,
        )
        rank_tolerance = (
            float(scientific.svd_rank_tolerance_multiplier)
            * np.finfo(np.dtype(scientific.svd_rank_tolerance_dtype)).eps
            * max(matrix.shape)
            * float(singular_values[0])
        )
        numerical_rank = int(np.sum(singular_values > rank_tolerance))
        if numerical_rank < int(
            scientific.minimum_numerical_rank_per_family
        ):
            raise ValueError(
                f"cluster-local {contrast_family} lacks configured numerical rank"
            )
        result = []
        max_components = min(
            int(scientific.maximum_components_per_family),
            numerical_rank,
        )
        total_energy = float(np.sum(np.square(singular_values)))
        for idx in range(max_components):
            sv = float(singular_values[idx])
            if not np.isfinite(sv) or sv <= 0.0:
                continue
            direction = _normalize_rows_configured(
                np.asarray(components[idx], dtype=dtype).reshape(1, -1),
                normalize=True,
                epsilon=float(scientific.normalization_epsilon),
                zero_vector_policy=scientific.zero_vector_policy,
                dtype=scientific.computation_dtype,
            )[0]
            name = f"{prefix}_pc{idx + 1}"
            result.append(
                {
                    "name": name,
                    "direction": direction,
                    "role": role,
                    "objective": objective,
                    "contrast_family": contrast_family,
                }
            )
            metadata.append(
                {
                    "outer_fold": int(outer_fold),
                    "name": name,
                    "role": role,
                    "objective": objective,
                    "contrast_family": contrast_family,
                    "cluster_component_index": int(idx + 1),
                    "singular_value": sv,
                    "explained_energy": (
                        float(sv**2 / total_energy) if total_energy > 0.0 else None
                    ),
                    "local_contrast_count": int(len(items)),
                    "svd_parameters": copy.deepcopy(svd_parameters),
                    "svd_sign_canonicalization_policy": (
                        scientific.svd_sign_canonicalization_policy
                    ),
                    "svd_rank_tolerance": float(rank_tolerance),
                    "svd_numerical_rank": numerical_rank,
                }
            )
        return result

    def _residualized_interaction_direction(
        self,
        patient_embeddings: np.ndarray,
        y: np.ndarray,
        t: np.ndarray,
        treatment_labels: np.ndarray,
        treatment_mask: np.ndarray,
        outcome_labels: np.ndarray,
        outcome_mask: np.ndarray,
        treatment_direction: Optional[np.ndarray],
        outcome_direction: Optional[np.ndarray],
        finite: np.ndarray,
    ) -> Optional[np.ndarray]:
        del y, t
        if treatment_direction is None or outcome_direction is None:
            return None
        base = finite & treatment_mask & outcome_mask
        treated_positive = base & (treatment_labels == 1) & (outcome_labels == 1)
        treated_negative = base & (treatment_labels == 1) & (outcome_labels == 0)
        untreated_positive = base & (treatment_labels == 0) & (outcome_labels == 1)
        untreated_negative = base & (treatment_labels == 0) & (outcome_labels == 0)
        if (
            min(
                int(np.sum(treated_positive)),
                int(np.sum(treated_negative)),
                int(np.sum(untreated_positive)),
                int(np.sum(untreated_negative)),
            )
            < 2
        ):
            return None
        raw = (
            np.mean(patient_embeddings[treated_positive], axis=0)
            - np.mean(patient_embeddings[treated_negative], axis=0)
            - np.mean(patient_embeddings[untreated_positive], axis=0)
            + np.mean(patient_embeddings[untreated_negative], axis=0)
        )
        residual = _residualize_vector_from_basis(raw, [treatment_direction, outcome_direction])
        if float(np.linalg.norm(residual)) <= 0.0:
            return None
        return residual

    def _chunk_similarity_features(
        self,
        generator: EmbeddingContrastEvidenceGenerator,
        positions: Sequence[int],
        direction: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        means = []
        maxes = []
        direction = _normalize_vector(np.asarray(direction, dtype=np.float32))
        for position in positions:
            chunks = generator._chunk_matrix(int(position))
            if chunks.size == 0:
                means.append(0.0)
                maxes.append(0.0)
                continue
            scores = np.asarray(chunks @ direction, dtype=float)
            finite = scores[np.isfinite(scores)]
            if len(finite) == 0:
                means.append(0.0)
                maxes.append(0.0)
            else:
                means.append(float(np.mean(finite)))
                maxes.append(float(np.max(finite)))
        return np.asarray(means, dtype=np.float32), np.asarray(maxes, dtype=np.float32)

    def _enabled_feature_discovery_methods(self) -> List[str]:
        methods = []
        if self._bow_enabled():
            methods.append("bow")
        if self._htr_enabled():
            methods.append("htr")
        if self._embedding_contrast_enabled():
            methods.append("embedding_contrast")
        return methods

    def _bow_enabled(self) -> bool:
        return bool(getattr(self.nn_config, "bow_discovery_enabled", True))

    def _htr_enabled(self) -> bool:
        return bool(getattr(self.nn_config, "htr_evidence_enabled", True))

    def _embedding_contrast_enabled(self) -> bool:
        return bool(getattr(self.nn_config.embedding_contrast, "enabled", False))

    def _matched_pair_uplift_enabled(self) -> bool:
        if str(self.config.outcome_type).lower() == "continuous":
            return False
        return bool(getattr(self.nn_config, "matched_pair_uplift_enabled", True))

    def _matched_pair_bow_enabled(self) -> bool:
        return (
            self._matched_pair_uplift_enabled()
            and self._bow_enabled()
            and bool(getattr(self.nn_config, "matched_pair_bow_enabled", True))
        )

    def _matched_pair_htr_enabled(self) -> bool:
        return (
            self._matched_pair_uplift_enabled()
            and self._htr_enabled()
            and bool(getattr(self.nn_config, "matched_pair_htr_enabled", True))
        )

    def _embedding_generator(self) -> EmbeddingContrastEvidenceGenerator:
        if self.embedding_evidence_generator is None:
            self.embedding_evidence_generator = EmbeddingContrastEvidenceGenerator(
                config=self.config,
                output_dir=self.artifact_dir,
                embedding_provider=self.embedding_provider,
            )
        return self.embedding_evidence_generator

    def _htr_provider(self) -> Any:
        if self.htr_evidence_provider is not None:
            if (
                self.htr_native_capture_sink is not None
                or self.matched_pair_native_capture_sink is not None
            ):
                raise RuntimeError(
                    "native HTR or matched-pair proof capture requires the genuine "
                    "default HTR provider"
                )
            return self.htr_evidence_provider
        if self._default_htr_provider is None:
            htr_num_workers = 0 if self._outer_backend_name() == "processes" else self.num_workers
            self._default_htr_provider = MultiModelForestStage1HTRProvider(
                config=self.config,
                output_dir=self.artifact_dir,
                device=self.device,
                gpu_ids=self.gpu_ids,
                num_workers=htr_num_workers,
            )
            self._default_htr_provider.native_capture_sink = self.htr_native_capture_sink
            self._default_htr_provider.native_pair_capture_sink = (
                self.matched_pair_native_capture_sink
            )
            self._default_htr_provider.checkpoint_store = self._active_checkpoint_store()
        return self._default_htr_provider

    def _sync_htr_fold_parallelism(self) -> None:
        htr_setting = self._htr_fold_parallelism_setting()
        avf_config = getattr(self.config.architecture, "agentic_attention_variable_forest", None)
        if avf_config is None:
            avf_config = AgenticAttentionVariableForestConfig()
            self.config.architecture.agentic_attention_variable_forest = avf_config
        avf_config.fold_parallelism = str(htr_setting)

    def _outer_backend_name(self) -> str:
        backend = str(
            getattr(self.nn_config, "outer_parallel_backend", "processes")
        ).strip().lower()
        if backend == "loky":
            backend = "processes"
        if backend not in {"threads", "processes"}:
            raise ValueError(
                "multi_model_forest.outer_parallel_backend must be "
                "'threads', 'processes', or 'loky'"
            )
        return backend

    def _bow_fold_parallelism_setting(self) -> str:
        setting = getattr(self.nn_config, "bow_fold_parallelism", None)
        if setting is None:
            setting = self.nn_config.fold_parallelism
        return str(setting).strip().lower()

    def _htr_fold_parallelism_setting(self) -> str:
        setting = getattr(self.nn_config, "htr_fold_parallelism", None)
        if setting is None:
            setting = self.nn_config.fold_parallelism
        return str(setting).strip().lower()

    def _parallel_n_jobs(self, setting: Any, tasks: int, *, auto_workers: int) -> int:
        if tasks <= 0:
            return 1
        setting_text = str(setting).strip().lower()
        if setting_text == "auto":
            return max(1, min(int(auto_workers), int(tasks)))
        return max(1, min(int(setting_text), int(tasks)))

    def _outer_n_jobs(self, folds: int) -> int:
        return self._parallel_n_jobs(
            self.nn_config.outer_parallelism,
            folds,
            auto_workers=self.num_workers,
        )

    def _inner_workers_for_outer_job(self, outer_n_jobs: int) -> int:
        if str(self.nn_config.fold_parallelism).strip().lower() != "auto":
            return self.num_workers
        return max(1, int(self.num_workers) // max(1, int(outer_n_jobs)))

    def _outer_devices(self, outer_n_jobs: int) -> List[torch.device]:
        if self.gpu_ids and self.device.type == "cuda":
            devices = [torch.device(f"cuda:{int(gpu_id)}") for gpu_id in self.gpu_ids]
            return devices[: max(1, min(len(devices), int(outer_n_jobs)))]
        return [self.device]

    def _fold_n_jobs(self, folds: int) -> int:
        # The proof sink owns in-memory references to the actual fitted sklearn
        # objects.  A process backend would copy the sink and silently discard
        # child-process captures; serialized capture mode is therefore
        # deliberately single-process and deterministic.
        if self.bow_native_capture_sink is not None:
            return 1
        return self._parallel_n_jobs(
            self._bow_fold_parallelism_setting(),
            folds,
            auto_workers=self.num_workers,
        )

    def _parallel_backend_name(self) -> str:
        return "loky" if self.nn_config.bow_parallel_backend == "processes" else "threading"

    def _run_fold_tasks(self, run_fold: Any, split_items: Sequence[Any]) -> List[Any]:
        n_jobs = self._fold_n_jobs(len(split_items))
        if n_jobs <= 1:
            return [
                run_fold(int(fold), np.asarray(fit_pos), np.asarray(heldout_pos))
                for fold, (fit_pos, heldout_pos) in split_items
            ]
        return Parallel(
            n_jobs=n_jobs,
            backend=self._parallel_backend_name(),
            batch_size=1,
            pre_dispatch="all",
        )(
            delayed(run_fold)(int(fold), np.asarray(fit_pos), np.asarray(heldout_pos))
            for fold, (fit_pos, heldout_pos) in split_items
        )

    def _save_outputs(self, results_df: pd.DataFrame) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_parquet(self.output_path, index=False)
        results_df.to_parquet(self.artifact_dir / "ite_estimates.parquet", index=False)
        metric_frame = pd.DataFrame(self.outer_metric_rows)
        metric_frame.to_csv(self.artifact_dir / "outer_cv_metrics.csv", index=False)
        _write_jsonl(self.artifact_dir / "feature_manifest.jsonl", self.feature_manifest_rows)
        _write_jsonl(
            self.artifact_dir / "embedding_contrast_feature_vectors.jsonl",
            self.embedding_feature_rows,
        )
        _write_jsonl(self.artifact_dir / "split_provenance.jsonl", self.split_provenance_rows)
        _write_jsonl(
            self.artifact_dir / "stage1_inner_model_evidence.jsonl",
            self.inner_model_evidence_rows,
        )
        if self.agentic_handoff_rows:
            handoff_path = self._agentic_handoff_path()
            _write_jsonl(handoff_path, self.agentic_handoff_rows)
            _write_json(
                handoff_path.with_suffix(".manifest.json"),
                {
                    "schema_version": "multi_model_agentic_discovery_handoff_v1",
                    "path": str(handoff_path),
                    "n_rows": int(len(self.agentic_handoff_rows)),
                    "scopes": sorted({str(row.get("scope")) for row in self.agentic_handoff_rows}),
                    "source": "stage1_primary_text_model_forest",
                },
            )
        if self.source_prediction_frames:
            pd.concat(self.source_prediction_frames, ignore_index=True).to_parquet(
                self.artifact_dir / "text_model_feature_predictions.parquet",
                index=False,
            )
        report = [
            "# Multi-Model Forest Stage 1 Text Models",
            "",
            f"- Rows: {len(self.dataset)}",
            f"- Outer folds: {len(self.outer_metric_rows)}",
            f"- Feature discovery methods: {', '.join(self._enabled_feature_discovery_methods())}",
            f"- Primary predictions: {self.output_path}",
            "- Agents used in primary forest: no",
            "- Stage 2 owner: ResearchAllEvidenceWorkflow/plain_handoff_stage2",
        ]
        (self.artifact_dir / "report.txt").write_text("\n".join(report) + "\n")
        logger.info("Multi-model stage1 forest predictions saved to: %s", self.output_path)

    def _agentic_handoff_path(self) -> Path:
        return self.artifact_dir / "agentic_handoff.jsonl"


def _vectorizer_params(view: BoWViewConfig) -> Dict[str, Any]:
    return _bow_vectorizer_params(view)


def _model_params(view: BoWViewConfig) -> Dict[str, Any]:
    return _bow_model_params(view)


def _bow_view_to_dict(view: BoWViewConfig) -> Dict[str, Any]:
    return asdict(view)


def _append_feature(
    train_cols: List[np.ndarray],
    test_cols: List[np.ndarray],
    names: List[str],
    feature_rows: List[Dict[str, Any]],
    *,
    train: np.ndarray,
    test: np.ndarray,
    name: str,
    role: str,
    source_family: str,
    outer_fold: int,
    objective: str,
    provenance: str,
    **metadata: Any,
) -> None:
    train_cols.append(np.asarray(train, dtype=np.float32))
    test_cols.append(np.asarray(test, dtype=np.float32))
    names.append(str(name))
    row = {
        "outer_fold": int(outer_fold),
        "feature_name": str(name),
        "feature_role": str(role),
        "source_family": str(source_family),
        "objective": str(objective),
        "provenance": str(provenance),
    }
    row.update({key: value for key, value in metadata.items() if value is not None})
    feature_rows.append(row)


def _column_matrix(cols: Sequence[np.ndarray], n_rows: int) -> np.ndarray:
    if not cols:
        return np.zeros((n_rows, 0), dtype=np.float32)
    return np.column_stack([np.asarray(col, dtype=np.float32).reshape(n_rows) for col in cols])


def _clean_train_test_matrices(
    train: np.ndarray, test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    train = np.asarray(train, dtype=np.float32)
    test = np.asarray(test, dtype=np.float32)
    if train.shape[1] == 0:
        return train, test
    means = np.nanmean(np.where(np.isfinite(train), train, np.nan), axis=0)
    means = np.where(np.isfinite(means), means, 0.0)
    train = np.where(np.isfinite(train), train, means)
    test = np.where(np.isfinite(test), test, means)
    return train.astype(np.float32, copy=False), test.astype(np.float32, copy=False)


def _source_prediction_frame(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    outer_fold: int,
    source_name: str,
    values: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    rows = []
    for split_role, frame, index in [
        ("train_inner_oof", train_df, 0),
        ("test_outer_train_fit", test_df, 1),
    ]:
        payload: Dict[str, Any] = {
            "_oci_row_id": frame["_oci_row_id"].to_numpy(),
            "outer_fold": int(outer_fold),
            "split_role": split_role,
            "source_name": str(source_name),
        }
        for column, pair in values.items():
            payload[column] = np.asarray(pair[index], dtype=float)
        rows.append(pd.DataFrame(payload))
    return pd.concat(rows, ignore_index=True)


def _source_prediction_architecture(source_name: Any) -> str:
    """Map a row-score producer onto the public Stage 1 architecture contract."""

    source = str(source_name).lower()
    if "matched_pair" in source or "pair_uplift" in source:
        return "matched_pair_uplift"
    if source.startswith("bow__") and "__nuisance" in source:
        return "bow_nuisance"
    if source.startswith("bow__") and "__effect" in source:
        return "bow_r_loss"
    if source.startswith("htr__"):
        return "htr_neural"
    if source.startswith("embedding__"):
        if "retrieval" in source or "tfidf" in source:
            return "tfidf_semantic_retrieval_contrasts"
        if "cluster" in source:
            return "embedding_clustered"
        return "embedding_whole_cohort"
    return "private_support"


def _weighted_binary_direction(
    embeddings: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    weights: Optional[np.ndarray],
) -> Tuple[Optional[np.ndarray], Dict[int, int]]:
    labels = np.asarray(labels, dtype=int)
    mask = np.asarray(mask, dtype=bool)
    pos = mask & (labels == 1)
    neg = mask & (labels == 0)
    counts = {1: int(np.sum(pos)), 0: int(np.sum(neg))}
    if counts[1] < 2 or counts[0] < 2:
        return None, counts
    if weights is None:
        return np.mean(embeddings[pos], axis=0) - np.mean(embeddings[neg], axis=0), counts
    weights = np.asarray(weights, dtype=float)
    pos_w = np.maximum(weights[pos], 0.0)
    neg_w = np.maximum(weights[neg], 0.0)
    pos_mean = (
        np.average(embeddings[pos], axis=0, weights=pos_w)
        if float(np.sum(pos_w)) > 0.0
        else np.mean(embeddings[pos], axis=0)
    )
    neg_mean = (
        np.average(embeddings[neg], axis=0, weights=neg_w)
        if float(np.sum(neg_w)) > 0.0
        else np.mean(embeddings[neg], axis=0)
    )
    return pos_mean - neg_mean, counts


def _finite_or_none(value: Any) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value
