"""Multi-model BoW-guided agentic variable discovery plus causal forest."""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import logging
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, fields
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.stats import chi2_contingency
from joblib import Parallel, delayed
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize as sklearn_normalize

from ..config import (
    AgenticFeatureSearchConfig,
    AppliedInferenceConfig,
    BoWViewConfig,
    ExplicitFeatureForestConfig,
    ExplicitFeatureSpec,
    MultiModelAgenticForestConfig,
    load_explicit_feature_specs_json,
    TfidfVectorizerScientificConfig,
)
from ..models.explicit_feature_featurizer import get_raw_explicit_features
from .agentic_explicit_feature_forest import (
    AgenticFeatureProposal,
    CausalForestExplicitEvaluator,
    SplitEvaluation,
    StructuredInteractionExplicitEvaluator,
    VLLMExplicitFeatureExtractionProvider,
    _clinical_text_examples,
    _get_agent_response_trace,
    _json_default,
    _normalize_feature_name,
    _safe_corr,
    _safe_roc_auc,
    _spec_to_dict,
    apply_agentic_alias_resolution,
    apply_agentic_value_harmonization,
    apply_proposals,
    make_explicit_feature_extraction_provider,
    make_feature_search_agent,
    validate_agentic_proposals,
)
from .embedding_contrast_discovery import (
    EmbeddingContrastEvidenceGenerator,
    redact_embedding_contrast_evidence,
)
from .discovery_randomness import derive_discovery_seed
from .agentic_attention_variable_forest import (
    AgenticAttentionVariableForestRunner,
    _attention_evidence_snippet,
    _attention_row_has_usable_text,
    _compact_token_spans,
    _parse_top_token_spans,
)

logger = logging.getLogger(__name__)


_DASH_TRANSLATION = dict.fromkeys(
    map(ord, "\u2010\u2011\u2012\u2013\u2014\u2212"),
    "-",
)

_AGENT_PROMPT_CONSENSUS_TOP_N = 40
_AGENT_PROMPT_VIEW_TOP_N = 12
_AGENT_PROMPT_EMBEDDING_CHUNKS_PER_TAIL = 3
_AGENT_PROMPT_EMBEDDING_CHUNK_CHARS = 600
_AGENT_PROMPT_CONCEPT_TOP_N = 8
_AGENT_PROMPT_HTR_ROWS_PER_STAGE = 36
_AGENT_PROMPT_HTR_SNIPPET_CHARS = 500
_AGENT_PROMPT_HTR_SUMMARY_CHARS = 320
_EVIDENCE_DIGEST_PROMPT_VERSION = "multi_model_agentic_evidence_digest_v1"
_EVIDENCE_DIGEST_ROLE_PROMPT_VERSION = "multi_model_agentic_evidence_digest_role_v1"
_EVIDENCE_DIGEST_BOW_ROWS_PER_LIST = 12
_EVIDENCE_DIGEST_EMBEDDING_CONTRASTS_PER_ROLE = 12
_EVIDENCE_DIGEST_TEXT_BLURBS_MAX = 80
_EVIDENCE_DIGEST_TEXT_BLURB_CHARS = 1800
_CONCEPT_INVENTORY_SCHEMA_VERSION = "multi_model_agentic_clustered_concept_inventory_v2"
_CONCEPT_CLUSTER_LABEL_PROMPT_VERSION = "multi_model_agentic_cluster_labeling_v2"
_CONCEPT_CLUSTER_MAX_AGENT_CLUSTERS = 120
_CONCEPT_CLUSTER_SNIPPET_CHARS = 320
_CONCEPT_CLUSTER_TOP_PHRASES = 12
_PARSIMONY_SCHEMA_VERSION = "multi_model_agentic_cluster_factor_parsimony_v2"
_PARSIMONY_FACTOR_PROMPT_VERSION = "multi_model_agentic_parsimony_factor_v1"


def _make_configured_explicit_evaluator(
    *,
    config: AppliedInferenceConfig,
    cf_config: ExplicitFeatureForestConfig,
) -> Any:
    """Select the configured structured head without changing legacy defaults.

    Older/standalone configurations do not necessarily contain the Stage 1
    ``multi_model_forest`` block.  Those continue to use the causal forest.
    When the block is present, its validated estimator selector is authoritative
    for both the ordinary and precomputed-discovery runners.
    """
    architecture = getattr(config, "architecture", None)
    stage1_config = getattr(architecture, "multi_model_forest", None)
    if stage1_config is None:
        return CausalForestExplicitEvaluator(config=config, cf_config=cf_config)

    estimator = getattr(stage1_config, "structured_effect_estimator", None)
    if estimator is None:
        return CausalForestExplicitEvaluator(config=config, cf_config=cf_config)
    estimator = str(estimator).strip().lower().replace("-", "_")
    if estimator in {"interaction", "s_learner", "interaction_s_learner"}:
        return StructuredInteractionExplicitEvaluator(
            config=config,
            cf_config=cf_config,
        )
    if estimator in {"causal_forest", "forest"}:
        return CausalForestExplicitEvaluator(config=config, cf_config=cf_config)
    raise ValueError(
        "multi_model_forest.structured_effect_estimator must be "
        "'interaction_s_learner' or 'causal_forest'"
    )


def _without_oracle_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the modeling frame with synthetic-oracle columns removed."""
    oracle_columns = [
        column
        for column in frame.columns
        if str(column).lower().startswith(("true_", "oracle_", "ground_truth"))
    ]
    return frame.drop(columns=oracle_columns, errors="ignore")


def run_multi_model_agentic_forest(
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    output_path: Path,
    device=None,
    gpu_ids: Optional[Sequence[int]] = None,
    num_workers: int = 1,
    proposal_agent: Optional[Any] = None,
    extraction_provider: Optional[Any] = None,
    evaluator: Optional[Any] = None,
    embedding_provider: Optional[Any] = None,
    htr_evidence_provider: Optional[Any] = None,
) -> None:
    """Run BoW-guided agentic variable discovery and final explicit-feature forest."""
    runner = MultiModelAgenticForestRunner(
        dataset=dataset,
        config=config,
        output_path=output_path,
        device=device,
        gpu_ids=gpu_ids,
        num_workers=num_workers,
        proposal_agent=proposal_agent,
        extraction_provider=extraction_provider,
        evaluator=evaluator,
        embedding_provider=embedding_provider,
        htr_evidence_provider=htr_evidence_provider,
    )
    runner.run()


def build_multi_model_agentic_discovery_handoff(
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    output_path: Path,
    device=None,
    gpu_ids: Optional[Sequence[int]] = None,
    num_workers: int = 1,
    *,
    include_candidate_consistency: bool = True,
) -> None:
    """Precompute agent-visible discovery evidence without running LLM agents."""
    runner = MultiModelAgenticForestRunner(
        dataset=dataset,
        config=config,
        output_path=Path(output_path).with_suffix(".predictions.placeholder.parquet"),
        device=device,
        gpu_ids=gpu_ids,
        num_workers=num_workers,
    )
    rows = runner.build_discovery_handoff_rows(
        include_candidate_consistency=include_candidate_consistency
    )
    output_path = Path(output_path)
    _write_jsonl(output_path, rows)
    scopes = sorted({str(row.get("scope")) for row in rows})
    _write_json(
        output_path.with_suffix(".manifest.json"),
        {
            "schema_version": "multi_model_agentic_discovery_handoff_v1",
            "path": str(output_path),
            "n_rows": int(len(rows)),
            "scopes": scopes,
            "include_candidate_consistency": bool(include_candidate_consistency),
        },
    )
    logger.info(
        "Saved multi-model agentic discovery handoff rows=%s path=%s",
        len(rows),
        output_path,
    )


def run_multi_model_agentic_forest_from_handoff(
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    output_path: Path,
    handoff_path: Path,
    device=None,
    gpu_ids: Optional[Sequence[int]] = None,
    num_workers: int = 1,
    proposal_agent: Optional[Any] = None,
    extraction_provider: Optional[Any] = None,
    evaluator: Optional[Any] = None,
    resume: bool = True,
) -> None:
    """Run the agentic explicit-feature branch from precomputed discovery evidence."""
    runner = PrecomputedDiscoveryMultiModelAgenticForestRunner(
        dataset=dataset,
        config=config,
        output_path=output_path,
        handoff_path=handoff_path,
        device=device,
        gpu_ids=gpu_ids,
        num_workers=num_workers,
        proposal_agent=proposal_agent,
        extraction_provider=extraction_provider,
        evaluator=evaluator,
        resume=resume,
    )
    runner.run()


class MultiModelHTREvidenceProvider:
    """Adapter that reuses the attention runner's HTR cross-fit stages."""

    def __init__(
        self,
        *,
        config: AppliedInferenceConfig,
        output_dir: Path,
        device: Optional[Any] = None,
        gpu_ids: Optional[Sequence[int]] = None,
        num_workers: int = 1,
    ) -> None:
        self.config = config
        self.output_dir = Path(output_dir)
        self.device = torch.device(device or "cpu")
        self.gpu_ids = list(gpu_ids) if gpu_ids is not None else None
        self.num_workers = 1 if num_workers is None else int(num_workers)
        self._runner: Optional[AgenticAttentionVariableForestRunner] = None

    def _ensure_runner(self, discovery_df: pd.DataFrame) -> AgenticAttentionVariableForestRunner:
        if self._runner is None:
            self._runner = AgenticAttentionVariableForestRunner(
                dataset=discovery_df,
                config=self.config,
                output_path=self.output_dir / "htr_evidence" / "predictions.parquet",
                device=self.device,
                gpu_ids=self.gpu_ids,
                num_workers=self.num_workers,
            )
        return self._runner

    def fit_nuisance(
        self,
        discovery_df: pd.DataFrame,
        outer_fold: int,
    ) -> Dict[str, Any]:
        runner = self._ensure_runner(discovery_df)
        return runner._crossfit_nuisance(discovery_df, outer_fold)

    def fit_effect(
        self,
        discovery_df: pd.DataFrame,
        nuisance_predictions: pd.DataFrame,
        outer_fold: int,
    ) -> Dict[str, Any]:
        runner = self._ensure_runner(discovery_df)
        return runner._crossfit_effect(discovery_df, nuisance_predictions, outer_fold)


class MultiModelAgenticForestRunner:
    """Sparse-text discovery path for explicit-variable causal forests."""

    def __init__(
        self,
        dataset: pd.DataFrame,
        config: AppliedInferenceConfig,
        output_path: Path,
        device: Optional[Any] = None,
        gpu_ids: Optional[Sequence[int]] = None,
        num_workers: int = 1,
        proposal_agent: Optional[Any] = None,
        extraction_provider: Optional[Any] = None,
        evaluator: Optional[Any] = None,
        embedding_provider: Optional[Any] = None,
        htr_evidence_provider: Optional[Any] = None,
        resume: bool = True,
    ) -> None:
        self.dataset = _without_oracle_columns(dataset).reset_index(drop=True).copy()
        self.dataset["_oci_row_id"] = np.arange(len(self.dataset), dtype=int)
        self.config = config
        self.output_path = Path(output_path)
        self.artifact_dir = self.output_path.parent / "multi_model_agentic_forest"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device or "cpu")
        self.gpu_ids = list(gpu_ids) if gpu_ids is not None else None
        self.num_workers = 1 if num_workers is None else int(num_workers)
        self._has_external_components = (
            proposal_agent is not None
            or extraction_provider is not None
            or evaluator is not None
            or embedding_provider is not None
            or htr_evidence_provider is not None
        )
        self._has_external_proposal_agent = proposal_agent is not None

        self.nn_config: MultiModelAgenticForestConfig = getattr(
            config.architecture,
            "multi_model_agentic_forest",
            MultiModelAgenticForestConfig(),
        )
        self.search_config: AgenticFeatureSearchConfig = getattr(
            config.architecture,
            "agentic_feature_search",
            AgenticFeatureSearchConfig(),
        )
        self.cf_config: ExplicitFeatureForestConfig = getattr(
            config.architecture,
            "explicit_feature_forest",
            ExplicitFeatureForestConfig(),
        )
        self.proposal_agent = proposal_agent or make_feature_search_agent(
            self.search_config
        )
        self.extraction_provider = (
            extraction_provider
            or make_explicit_feature_extraction_provider(
                config=config,
                output_dir=self.artifact_dir,
            )
        )
        self.evaluator = (
            evaluator
            if evaluator is not None
            else _make_configured_explicit_evaluator(
                config=config,
                cf_config=self.cf_config,
            )
        )
        self.embedding_provider = embedding_provider
        self.htr_evidence_provider = htr_evidence_provider
        self.resume = bool(resume)
        self.embedding_evidence_generator: Optional[EmbeddingContrastEvidenceGenerator] = None
        self._default_htr_evidence_provider: Optional[MultiModelHTREvidenceProvider] = None
        self._concept_cluster_embedding_encoder: Optional[Any] = None
        self._concept_cluster_embedding_encoder_key: Optional[Tuple[str, str, str]] = None

        self.bow_prediction_frames: List[pd.DataFrame] = []
        self.htr_nuisance_prediction_frames: List[pd.DataFrame] = []
        self.htr_effect_prediction_frames: List[pd.DataFrame] = []
        self.ensemble_nuisance_prediction_frames: List[pd.DataFrame] = []
        self.htr_attention_rows: List[Dict[str, Any]] = []
        self.importance_rows: List[Dict[str, Any]] = []
        self.embedding_evidence_rows: List[Dict[str, Any]] = []
        self.agent_rows: List[Dict[str, Any]] = []
        self.extracted_feature_diagnostic_rows: List[Dict[str, Any]] = []
        self.candidate_signal_review_rows: List[Dict[str, Any]] = []
        self.parsimony_review_rows: List[Dict[str, Any]] = []
        self.parsimony_cluster_rows: List[Dict[str, Any]] = []
        self.parsimony_factor_rows: List[Dict[str, Any]] = []
        self.parsimony_evaluation_rows: List[Dict[str, Any]] = []
        self.feature_set_rows: List[Dict[str, Any]] = []
        self.outer_metric_rows: List[Dict[str, Any]] = []
        self.split_provenance_rows: List[Dict[str, Any]] = []
        self.prediction_results: Optional[pd.DataFrame] = None
        self.alias_reference_specs: List[ExplicitFeatureSpec] = self._initial_specs()
        self.low_coverage_review_candidates_by_outer: Dict[int, Dict[str, Dict[str, Any]]] = {}

    def build_discovery_handoff_rows(
        self,
        *,
        include_candidate_consistency: bool = True,
    ) -> List[Dict[str, Any]]:
        """Build fold-level discovery evidence rows for a later agent-only run."""
        logger.info("=" * 80)
        logger.info("MULTI-MODEL AGENTIC DISCOVERY HANDOFF")
        logger.info("=" * 80)
        self._validate_required_evidence_sources()
        self._ensure_prespecified_features()

        splits = self._analysis_splits()
        if self._embedding_contrast_enabled() and self.embedding_provider is None:
            self._embedding_contrast_generator().prepare(self.dataset)

        rows: List[Dict[str, Any]] = []
        for outer_fold, train_idx, _test_idx in splits:
            discovery_df = self.dataset.iloc[train_idx].reset_index(drop=True)
            logger.info(
                "Precomputing agentic discovery handoff outer_fold=%s rows=%s",
                outer_fold,
                len(discovery_df),
            )
            result = self._fit_bow_discovery(discovery_df, int(outer_fold))
            rows.append(
                _agentic_discovery_handoff_row(
                    result,
                    fold_key=int(outer_fold),
                    outer_fold=int(outer_fold),
                    scope="full_outer_train",
                    n_rows=len(discovery_df),
                )
            )
            if include_candidate_consistency and bool(
                getattr(self.nn_config, "candidate_consistency_enabled", True)
            ):
                rows.extend(
                    self._build_inner_discovery_handoff_rows(
                        outer_fold=int(outer_fold),
                        discovery_df=discovery_df,
                    )
                )
        return rows

    def _build_inner_discovery_handoff_rows(
        self,
        *,
        outer_fold: int,
        discovery_df: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        try:
            fold_count = _bounded_fold_count(
                int(self.nn_config.candidate_consistency_inner_folds),
                len(discovery_df),
            )
        except ValueError:
            return []

        splitter = KFold(
            n_splits=fold_count,
            shuffle=True,
            random_state=51_000 + int(outer_fold),
        )
        rows: List[Dict[str, Any]] = []
        for inner_fold, (fit_pos, heldout_pos) in enumerate(
            splitter.split(discovery_df),
            start=1,
        ):
            fit_pos = np.asarray(fit_pos, dtype=int)
            heldout_pos = np.asarray(heldout_pos, dtype=int)
            fold_key = 1000 * int(outer_fold) + int(inner_fold)
            inner_df = discovery_df.iloc[fit_pos].reset_index(drop=True)
            logger.info(
                "Precomputing agentic discovery handoff outer_fold=%s "
                "inner_fold=%s rows=%s heldout_rows=%s",
                outer_fold,
                inner_fold,
                len(inner_df),
                len(heldout_pos),
            )
            result = self._fit_bow_discovery(inner_df, outer_fold=fold_key)
            rows.append(
                _agentic_discovery_handoff_row(
                    result,
                    fold_key=fold_key,
                    outer_fold=int(outer_fold),
                    scope="candidate_consistency_inner_train",
                    n_rows=len(inner_df),
                    inner_fold=int(inner_fold),
                    heldout_rows=len(heldout_pos),
                )
            )
        return rows

    def run(self) -> None:
        try:
            self._run_impl()
        finally:
            self._release_concept_cluster_embedding_encoder()

    def _run_impl(self) -> None:
        logger.info("=" * 80)
        logger.info("MULTI-MODEL AGENTIC FEATURE CAUSAL FOREST")
        logger.info("=" * 80)
        self._validate_required_evidence_sources()
        self._ensure_prespecified_features()

        splits = self._analysis_splits()
        self.split_provenance_rows = self._split_provenance_rows(splits)
        if self._embedding_contrast_enabled() and self.embedding_provider is None:
            self._embedding_contrast_generator().prepare(self.dataset)
        prediction_frames: List[pd.DataFrame] = []
        pending_splits: List[Tuple[int, np.ndarray, np.ndarray]] = []
        if self.resume:
            for outer_fold, train_idx, test_idx in splits:
                cached = self._load_outer_fold_checkpoint(
                    outer_fold=int(outer_fold),
                    expected_prediction_rows=len(test_idx),
                )
                if cached is None:
                    pending_splits.append((outer_fold, train_idx, test_idx))
                    continue
                logger.info(
                    "Resuming multi-model agentic fold %s from checkpoint",
                    outer_fold,
                )
                prediction_frames.append(cached["predictions"])
                self._extend_from_fold_result(cached)
        else:
            pending_splits = splits

        outer_n_jobs = self._outer_n_jobs(len(pending_splits))
        if outer_n_jobs > 1 and self._htr_evidence_enabled():
            logger.warning(
                "Outer fold parallelism disabled because integrated HTR evidence "
                "loads neural models; use BoW fold parallelism or disable HTR with "
                "a documented reason for lightweight runs."
            )
            outer_n_jobs = 1
        if outer_n_jobs > 1 and self._has_external_components:
            logger.warning(
                "Outer fold parallelism disabled because custom agent/extractor/"
                "evaluator objects were supplied and may not be thread-safe."
            )
            outer_n_jobs = 1

        if outer_n_jobs > 1:
            backend = self._parallel_backend_name()
            logger.info(
                "Running %s multi-model outer fold(s) with outer_parallelism=%s "
                "backend=%s joblib_backend=%s",
                len(pending_splits),
                outer_n_jobs,
                self.nn_config.bow_parallel_backend,
                backend,
            )
            fold_results = Parallel(
                n_jobs=outer_n_jobs,
                backend=backend,
                batch_size=1,
                pre_dispatch="all",
            )(
                delayed(_run_multi_model_outer_fold_worker)(
                    self.dataset,
                    self.config,
                    self.artifact_dir,
                    int(outer_fold),
                    np.asarray(train_idx),
                    np.asarray(test_idx),
                    self._inner_workers_for_outer_job(outer_n_jobs),
                )
                for outer_fold, train_idx, test_idx in pending_splits
            )
            fold_results = sorted(fold_results, key=lambda item: item["outer_fold"])
            for item in fold_results:
                prediction_frames.append(item["predictions"])
                self._extend_from_fold_result(item)
        else:
            for outer_fold, train_idx, test_idx in pending_splits:
                logger.info(
                    "Multi-model agentic fold %s: train=%s test=%s",
                    outer_fold,
                    len(train_idx),
                    len(test_idx),
                )
                predictions = self._run_one_analysis_split(
                    outer_fold=outer_fold,
                    train_idx=train_idx,
                    test_idx=test_idx,
                )
                prediction_frames.append(predictions)
                self._save_outer_fold_checkpoint(
                    outer_fold=int(outer_fold),
                    predictions=predictions,
                    target_dir=(self.artifact_dir / f"outer_fold_{int(outer_fold):03d}"),
                )

        results_df = pd.concat(prediction_frames).sort_values("_oci_row_id")
        self._save_predictions(results_df)
        self._save_artifacts()

    def _extend_from_fold_result(self, item: Dict[str, Any]) -> None:
        self.bow_prediction_frames.extend(item.get("bow_prediction_frames", []))
        self.htr_nuisance_prediction_frames.extend(
            item.get("htr_nuisance_prediction_frames", [])
        )
        self.htr_effect_prediction_frames.extend(
            item.get("htr_effect_prediction_frames", [])
        )
        self.ensemble_nuisance_prediction_frames.extend(
            item.get("ensemble_nuisance_prediction_frames", [])
        )
        self.htr_attention_rows.extend(item.get("htr_attention_rows", []))
        self.importance_rows.extend(item.get("importance_rows", []))
        self.embedding_evidence_rows.extend(item.get("embedding_evidence_rows", []))
        self.agent_rows.extend(item.get("agent_rows", []))
        self.extracted_feature_diagnostic_rows.extend(
            item.get("extracted_feature_diagnostic_rows", [])
        )
        self.candidate_signal_review_rows.extend(
            item.get("candidate_signal_review_rows", [])
        )
        self.parsimony_review_rows.extend(item.get("parsimony_review_rows", []))
        self.parsimony_cluster_rows.extend(item.get("parsimony_cluster_rows", []))
        self.parsimony_factor_rows.extend(item.get("parsimony_factor_rows", []))
        self.parsimony_evaluation_rows.extend(item.get("parsimony_evaluation_rows", []))
        self.feature_set_rows.extend(item.get("feature_set_rows", []))
        self.outer_metric_rows.extend(item.get("outer_metric_rows", []))

    def _load_outer_fold_checkpoint(
        self,
        *,
        outer_fold: int,
        expected_prediction_rows: int,
    ) -> Optional[Dict[str, Any]]:
        fold = int(outer_fold)
        target_dir = self.artifact_dir / f"outer_fold_{fold:03d}"
        prediction_path = target_dir / "predictions.parquet"
        summary_path = target_dir / "checkpoint_summary.json"
        if not prediction_path.exists() or not summary_path.exists():
            return None
        try:
            summary = _read_json(summary_path)
            predictions = pd.read_parquet(prediction_path)
        except Exception as exc:
            logger.warning(
                "Ignoring unreadable multi-model agentic fold checkpoint %s: %s",
                target_dir,
                exc,
            )
            return None
        if int(summary.get("outer_fold", fold)) != fold:
            return None
        if len(predictions) != int(expected_prediction_rows):
            logger.warning(
                "Ignoring incomplete multi-model agentic fold checkpoint %s: "
                "predictions=%s expected=%s",
                target_dir,
                len(predictions),
                expected_prediction_rows,
            )
            return None
        metric_rows = _read_csv_records(target_dir / "outer_cv_metrics.csv")
        selected_rows = _read_json(target_dir / "selected_feature_sets.json", default=[])
        agent_rows = _read_jsonl(target_dir / "agent_candidate_proposals.jsonl")
        if self._concept_inventory_enabled():
            consistency_enabled = bool(
                getattr(self.nn_config, "candidate_consistency_enabled", True)
            )
            if not any(
                _agent_row_has_concept_inventory(
                    row,
                    consistency_enabled=consistency_enabled,
                )
                for row in agent_rows
            ):
                logger.info(
                    "Ignoring multi-model agentic fold checkpoint without concept "
                    "inventory outer_fold=%s path=%s",
                    fold,
                    target_dir,
                )
                return None
        parsimony_rows = _read_jsonl(target_dir / "parsimony_review_by_fold.jsonl")
        if bool(getattr(self.nn_config, "parsimony_review_enabled", False)) and not any(
            isinstance(row, dict)
            and row.get("schema_version") == _PARSIMONY_SCHEMA_VERSION
            for row in parsimony_rows
        ):
            logger.info(
                "Ignoring multi-model agentic fold checkpoint with legacy parsimony "
                "schema outer_fold=%s path=%s",
                fold,
                target_dir,
            )
            return None
        return {
            "outer_fold": fold,
            "predictions": predictions,
            "bow_prediction_frames": [],
            "htr_nuisance_prediction_frames": [],
            "htr_effect_prediction_frames": [],
            "ensemble_nuisance_prediction_frames": [],
            "htr_attention_rows": [],
            "importance_rows": _read_jsonl(target_dir / "text_evidence.bow.jsonl"),
            "embedding_evidence_rows": _read_jsonl(
                target_dir / "text_evidence.embedding.jsonl"
            ),
            "agent_rows": agent_rows,
            "extracted_feature_diagnostic_rows": _read_jsonl(
                target_dir / "extracted_feature_diagnostics_by_fold.jsonl"
            ),
            "candidate_signal_review_rows": _read_jsonl(
                target_dir / "candidate_signal_review.jsonl"
            ),
            "parsimony_review_rows": parsimony_rows,
            "parsimony_cluster_rows": _read_jsonl(
                target_dir / "parsimony_clusters_by_fold.jsonl"
            ),
            "parsimony_factor_rows": _read_jsonl(
                target_dir / "parsimony_factor_proposals_by_fold.jsonl"
            ),
            "parsimony_evaluation_rows": _read_jsonl(
                target_dir / "parsimony_replacement_evaluations_by_fold.jsonl"
            ),
            "feature_set_rows": selected_rows if isinstance(selected_rows, list) else [],
            "outer_metric_rows": metric_rows,
        }

    def _run_one_analysis_split_isolated(
        self,
        outer_fold: int,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        outer_n_jobs: int,
    ) -> Dict[str, Any]:
        logger.info(
            "Multi-model agentic isolated fold %s: train=%s test=%s",
            outer_fold,
            len(train_idx),
            len(test_idx),
        )
        fold_runner = MultiModelAgenticForestRunner(
            dataset=self.dataset,
            config=self.config,
            output_path=(
                self.artifact_dir / f"outer_fold_{int(outer_fold):03d}" / "predictions.parquet"
            ),
            num_workers=self._inner_workers_for_outer_job(outer_n_jobs),
        )
        try:
            predictions = fold_runner._run_one_analysis_split(
                outer_fold=outer_fold,
                train_idx=train_idx,
                test_idx=test_idx,
            )
            fold_runner._save_outer_fold_checkpoint(
                outer_fold=int(outer_fold),
                predictions=predictions,
                target_dir=self.artifact_dir / f"outer_fold_{int(outer_fold):03d}",
            )
            return {
                "outer_fold": int(outer_fold),
                "predictions": predictions,
                "bow_prediction_frames": fold_runner.bow_prediction_frames,
                "htr_nuisance_prediction_frames": fold_runner.htr_nuisance_prediction_frames,
                "htr_effect_prediction_frames": fold_runner.htr_effect_prediction_frames,
                "ensemble_nuisance_prediction_frames": (
                    fold_runner.ensemble_nuisance_prediction_frames
                ),
                "htr_attention_rows": fold_runner.htr_attention_rows,
                "importance_rows": fold_runner.importance_rows,
                "embedding_evidence_rows": fold_runner.embedding_evidence_rows,
                "agent_rows": fold_runner.agent_rows,
                "extracted_feature_diagnostic_rows": (
                    fold_runner.extracted_feature_diagnostic_rows
                ),
                "candidate_signal_review_rows": fold_runner.candidate_signal_review_rows,
                "parsimony_review_rows": fold_runner.parsimony_review_rows,
                "parsimony_cluster_rows": fold_runner.parsimony_cluster_rows,
                "parsimony_factor_rows": fold_runner.parsimony_factor_rows,
                "parsimony_evaluation_rows": fold_runner.parsimony_evaluation_rows,
                "feature_set_rows": fold_runner.feature_set_rows,
                "outer_metric_rows": fold_runner.outer_metric_rows,
            }
        finally:
            fold_runner._release_concept_cluster_embedding_encoder()

    def _analysis_splits(self) -> List[Tuple[int, np.ndarray, np.ndarray]]:
        if self.config.cv_folds > 1:
            splits = KFold(
                n_splits=self.config.cv_folds,
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

        all_idx = np.arange(len(self.dataset))
        if bool(getattr(self.nn_config, "require_honest_outer_split", False)):
            raise ValueError(
                "multi_model_agentic_forest.require_honest_outer_split=True "
                "requires cv_folds > 1 or a split column with a held-out 'test' split"
            )
        logger.warning(
            "No held-out split configured for multi_model_agentic_forest; "
            "variable discovery and final estimates will use the full dataset."
        )
        return [(1, all_idx, all_idx)]

    def _split_provenance_rows(
        self,
        splits: Sequence[Tuple[int, np.ndarray, np.ndarray]],
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        if self.config.cv_folds > 1:
            split_source = "kfold_cv"
        elif self.config.split_column in self.dataset.columns and "test" in set(
            self.dataset[self.config.split_column]
        ):
            split_source = "configured_split_column"
        else:
            split_source = "full_data_refit"
        for outer_fold, train_idx, test_idx in splits:
            train_idx = np.asarray(train_idx, dtype=int)
            test_idx = np.asarray(test_idx, dtype=int)
            honest = _split_is_honest(train_idx, test_idx)
            rows.append(
                {
                    "outer_fold": int(outer_fold),
                    "split_source": split_source,
                    "train_rows": int(len(train_idx)),
                    "test_rows": int(len(test_idx)),
                    "train_row_ids": train_idx.astype(int).tolist(),
                    "test_row_ids": test_idx.astype(int).tolist(),
                    "honest_outer_holdout": bool(honest),
                    "estimation_provenance": (
                        "honest_outer_fold" if honest else "full_data_refit_non_honest"
                    ),
                }
            )
        return rows

    def _run_one_analysis_split(
        self,
        outer_fold: int,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
    ) -> pd.DataFrame:
        self._ensure_prespecified_features()
        discovery_df = self.dataset.iloc[train_idx].reset_index(drop=True)
        bow_result = self._fit_bow_discovery(discovery_df, outer_fold)
        non_htr_predictions = _non_htr_prediction_frame(bow_result["predictions"])
        if not non_htr_predictions.empty:
            self.bow_prediction_frames.append(non_htr_predictions)
        artifact_context = self._artifact_agent_context(bow_result["context"])
        for view in bow_result["importance"].get("views", []) or []:
            feature_importance = {
                key: value
                for key, value in view.items()
                if key not in {"view_name", "view_index", "view_config", "metrics"}
            }
            self.importance_rows.append(
                {
                    "record_type": "view",
                    "outer_fold": int(outer_fold),
                    "view_index": int(view.get("view_index", -1)),
                    "view_name": view.get("view_name"),
                    "view_config": view.get("view_config"),
                    "metrics": view.get("metrics"),
                    "feature_importance": feature_importance,
                }
            )
        ensemble_importance = bow_result["importance"].get("ensemble_r")
        if isinstance(ensemble_importance, dict):
            for view in ensemble_importance.get("views", []) or []:
                feature_importance = {
                    key: value
                    for key, value in view.items()
                    if key not in {"view_name", "view_index", "view_config", "metrics"}
                }
                self.importance_rows.append(
                    {
                        "record_type": "ensemble_r_view",
                        "outer_fold": int(outer_fold),
                        "view_index": int(view.get("view_index", -1)),
                        "view_name": view.get("view_name"),
                        "view_config": view.get("view_config"),
                        "metrics": view.get("metrics"),
                        "feature_importance": feature_importance,
                    }
                )
            self.importance_rows.append(
                {
                    "record_type": "ensemble_r_consensus",
                    "outer_fold": int(outer_fold),
                    "phrase_consensus": ensemble_importance.get("phrase_consensus", []),
                }
            )
        self.importance_rows.append(
            {
                "record_type": "consensus",
                "outer_fold": int(outer_fold),
                "phrase_consensus": bow_result["importance"].get("phrase_consensus", []),
                "context": artifact_context,
            }
        )
        embedding_evidence = bow_result.get("embedding_contrast_evidence") or {}
        if embedding_evidence:
            self.embedding_evidence_rows.append(
                {
                    "outer_fold": int(outer_fold),
                    "embedding_contrast_evidence": (
                        embedding_evidence
                        if self.search_config.save_agent_context
                        else redact_embedding_contrast_evidence(embedding_evidence)
                    ),
                }
            )

        selected_specs = self._propose_selected_specs(
            outer_fold=outer_fold,
            discovery_df=discovery_df,
            bow_context=bow_result["context"],
        )

        self._validate_complete_document_extraction(selected_specs)
        self.dataset = self.extraction_provider.ensure_features(
            self.dataset,
            selected_specs,
        )
        train_df = self.dataset.iloc[train_idx].copy()
        test_df = self.dataset.iloc[test_idx].copy()
        selected_specs = self._filter_specs_by_extraction_coverage(
            train_df=train_df,
            specs=selected_specs,
            outer_fold=outer_fold,
        )
        review_result = self._review_extracted_features_if_needed(
            outer_fold=outer_fold,
            train_idx=train_idx,
            selected_specs=selected_specs,
            bow_result=bow_result,
            embedding_evidence=embedding_evidence,
        )
        selected_specs = review_result["selected_specs"]
        train_df = self.dataset.iloc[train_idx].copy()
        test_df = self.dataset.iloc[test_idx].copy()
        parsimony_result = self._run_mandatory_parsimony_review(
            outer_fold=outer_fold,
            train_idx=train_idx,
            selected_specs=selected_specs,
            bow_result=bow_result,
            embedding_evidence=embedding_evidence,
        )
        selected_specs = parsimony_result["selected_specs"]
        train_df = self.dataset.iloc[train_idx].copy()
        test_df = self.dataset.iloc[test_idx].copy()
        signal_review_rows = self._build_candidate_signal_review_rows(
            outer_fold=outer_fold,
            train_df=train_df,
            selected_specs=selected_specs,
            bow_result=bow_result,
            embedding_evidence=embedding_evidence,
        )
        self.candidate_signal_review_rows.extend(signal_review_rows)
        self.feature_set_rows.append(
            {
                "outer_fold": int(outer_fold),
                "selected_features": [_spec_to_dict(spec) for spec in selected_specs],
                "confounders": [spec.name for spec in selected_specs if "confounder" in spec.roles],
                "effect_modifiers": [
                    spec.name for spec in selected_specs if "effect_modifier" in spec.roles
                ],
                "extracted_feature_review": review_result["summary"],
                "parsimony_review": parsimony_result["summary"],
            }
        )

        final_eval: SplitEvaluation = self.evaluator.evaluate_split(
            train_df=_without_oracle_columns(train_df),
            test_df=_without_oracle_columns(test_df),
            specs=selected_specs,
            fold_id=outer_fold,
        )
        predictions = final_eval.predictions.copy()
        honest = _split_is_honest(train_idx, test_idx)
        predictions["outer_fold"] = int(outer_fold)
        predictions["honest_outer_holdout"] = bool(honest)
        predictions["estimation_provenance"] = (
            "honest_outer_fold" if honest else "full_data_refit_non_honest"
        )
        predictions["selected_feature_names"] = ",".join(spec.name for spec in selected_specs)
        predictions["selected_feature_roles"] = _format_selected_feature_roles(selected_specs)
        predictions["selected_confounder_names"] = ",".join(
            spec.name for spec in selected_specs if "confounder" in spec.roles
        )
        predictions["selected_effect_modifier_names"] = ",".join(
            spec.name for spec in selected_specs if "effect_modifier" in spec.roles
        )

        self.outer_metric_rows.append(
            {
                "outer_fold": int(outer_fold),
                "honest_outer_holdout": bool(honest),
                "estimation_provenance": (
                    "honest_outer_fold" if honest else "full_data_refit_non_honest"
                ),
                "n_selected_features": int(len(selected_specs)),
                **_prefix_metrics(
                    "extracted_feature_review_",
                    _scalar_metrics(review_result["summary"]),
                ),
                **_prefix_metrics(
                    "parsimony_review_",
                    _scalar_metrics(parsimony_result["summary"]),
                ),
                **_scalar_metrics(final_eval.metrics),
                **_prefix_metrics("bow_", bow_result["metrics"]),
            }
        )
        return predictions

    def _fit_bow_discovery(
        self,
        discovery_df: pd.DataFrame,
        outer_fold: int,
    ) -> Dict[str, Any]:
        texts = _normalize_texts(discovery_df[self.config.text_column].fillna(""))
        y = discovery_df[self.config.outcome_column].to_numpy(dtype=float)
        t = discovery_df[self.config.treatment_column].to_numpy(dtype=float)
        prespecified_specs = self._initial_specs()
        explicit_feature_dicts = _columns_to_feature_dicts(
            discovery_df,
            prespecified_specs,
        )

        view_results: List[Dict[str, Any]] = []
        if self._bow_discovery_enabled():
            for view_index, view in enumerate(self.nn_config.bow_views):
                view_results.append(
                    self._fit_one_bow_view(
                        discovery_df=discovery_df,
                        texts=texts,
                        y=y,
                        t=t,
                        outer_fold=outer_fold,
                        view=view,
                        view_index=view_index,
                        explicit_feature_dicts=explicit_feature_dicts,
                        explicit_specs=prespecified_specs,
                    )
                )
        else:
            logger.info(
                "Outer fold %s BoW discovery disabled; skipping sparse text views",
                outer_fold,
            )

        htr_nuisance_result = self._fit_htr_nuisance_discovery(
            discovery_df,
            outer_fold,
        )
        htr_evidence = None
        if htr_nuisance_result is not None:
            self.htr_nuisance_prediction_frames.append(htr_nuisance_result["predictions"])
            self.htr_attention_rows.extend(htr_nuisance_result.get("attention", []))
            htr_evidence = {
                "nuisance": {
                    "metrics": htr_nuisance_result.get("metrics", {}),
                    "attention": htr_nuisance_result.get("attention", []),
                }
            }

        nuisance_results = (
            [*view_results, htr_nuisance_result]
            if htr_nuisance_result is not None
            else list(view_results)
        )
        ensemble_result = self._fit_ensemble_r_discovery(
            discovery_df=discovery_df,
            texts=texts,
            y=y,
            t=t,
            outer_fold=outer_fold,
            view_results=view_results,
            nuisance_results=nuisance_results,
            explicit_feature_dicts=explicit_feature_dicts,
            explicit_specs=prespecified_specs,
        )
        if ensemble_result is not None:
            self.ensemble_nuisance_prediction_frames.append(ensemble_result["nuisance_predictions"])

        if ensemble_result is not None and htr_nuisance_result is not None:
            htr_effect_result = self._fit_htr_effect_discovery(
                discovery_df=discovery_df,
                outer_fold=outer_fold,
                nuisance_predictions=ensemble_result["nuisance_predictions"],
            )
            if htr_effect_result is not None:
                ensemble_result["htr_effect_result"] = htr_effect_result
                self.htr_effect_prediction_frames.append(htr_effect_result["predictions"])
                self.htr_attention_rows.extend(htr_effect_result.get("attention", []))
                assert htr_evidence is not None
                htr_evidence["effect"] = {
                    "metrics": htr_effect_result.get("metrics", {}),
                    "attention": htr_effect_result.get("attention", []),
                }

        prediction_frames = [result["predictions"] for result in view_results]
        if htr_nuisance_result is not None:
            prediction_frames.append(htr_nuisance_result["predictions"])
        if ensemble_result is not None:
            prediction_frames.extend(
                result["predictions"] for result in ensemble_result.get("view_results", [])
            )
            htr_effect_result = ensemble_result.get("htr_effect_result")
            if htr_effect_result is not None:
                prediction_frames.append(htr_effect_result["predictions"])
        predictions = (
            pd.concat(prediction_frames, ignore_index=True)
            if prediction_frames
            else pd.DataFrame(columns=["_oci_row_id", "outer_fold"])
        )
        metrics = _multi_view_metrics(view_results)
        metrics["feature_discovery_methods"] = self._enabled_feature_discovery_methods()
        if htr_nuisance_result is not None:
            metrics["htr_nuisance"] = htr_nuisance_result.get("metrics", {})
        if ensemble_result is not None:
            metrics["ensemble_r"] = ensemble_result["metrics"]
            if ensemble_result.get("htr_effect_result") is not None:
                metrics["htr_effect"] = ensemble_result["htr_effect_result"].get(
                    "metrics",
                    {},
                )
            for key, value in _scalar_metrics(ensemble_result["metrics"]).items():
                metrics[f"ensemble_{key}"] = value
        importance = _multi_view_importance(
            view_results,
            top_n=int(self.nn_config.top_n_features),
        )
        importance["feature_discovery_methods"] = self._enabled_feature_discovery_methods()
        if ensemble_result is not None:
            importance["ensemble_r"] = ensemble_result["importance"]

        pseudo_targets = [result["pseudo_target"] for result in view_results]
        t_resids = [result["t_resid"] for result in view_results]
        pseudo_target_names = [
            str(result.get("view_name") or getattr(result.get("view"), "name", "view"))
            for result in view_results
        ]
        if ensemble_result is not None:
            pseudo_targets.append(ensemble_result["pseudo_target"])
            t_resids.append(ensemble_result["t_resid"])
            pseudo_target_names.append(
                str(ensemble_result.get("target_source") or "ensemble_mean_nuisance")
            )
        embedding_evidence = self._build_embedding_contrast_evidence(
            discovery_df=discovery_df,
            y=y,
            t=t,
            pseudo_target=pseudo_targets,
            t_resid=t_resids,
            pseudo_target_names=pseudo_target_names,
            importance=importance,
        )
        context = self._build_agent_context(
            outer_fold=outer_fold,
            discovery_df=discovery_df,
            metrics=metrics,
            importance=importance,
            embedding_evidence=embedding_evidence,
            htr_evidence=htr_evidence,
        )
        return {
            "predictions": predictions,
            "metrics": metrics,
            "importance": importance,
            "embedding_contrast_evidence": embedding_evidence,
            "htr_evidence": htr_evidence or {},
            "context": context,
        }

    def _fit_one_bow_view(
        self,
        *,
        discovery_df: pd.DataFrame,
        texts: Sequence[str],
        y: np.ndarray,
        t: np.ndarray,
        outer_fold: int,
        view: BoWViewConfig,
        view_index: int,
        explicit_feature_dicts: Optional[List[Dict[str, Any]]] = None,
        explicit_specs: Optional[List[ExplicitFeatureSpec]] = None,
    ) -> Dict[str, Any]:
        e_hat = self._crossfit_binary(
            texts,
            t,
            "treatment",
            outer_fold,
            view=view,
            view_index=view_index,
            explicit_feature_dicts=explicit_feature_dicts,
            explicit_specs=explicit_specs,
        )
        if self.config.outcome_type == "continuous":
            m_hat = self._crossfit_continuous(
                texts,
                y,
                "outcome",
                outer_fold,
                view=view,
                view_index=view_index,
                explicit_feature_dicts=explicit_feature_dicts,
                explicit_specs=explicit_specs,
            )
        else:
            m_hat = self._crossfit_binary(
                texts,
                y,
                "outcome",
                outer_fold,
                view=view,
                view_index=view_index,
                explicit_feature_dicts=explicit_feature_dicts,
                explicit_specs=explicit_specs,
            )

        e_clipped = np.clip(e_hat, self.nn_config.e_clip, 1.0 - self.nn_config.e_clip)
        t_resid = t - e_clipped
        y_resid = y - m_hat
        pseudo_target = y_resid / t_resid

        tau_hat = self._crossfit_pseudo_target(
            texts,
            pseudo_target,
            t_resid**2,
            outer_fold,
            dina_labels=(t, y) if self.config.outcome_type == "binary" else None,
            view=view,
            view_index=view_index,
            explicit_feature_dicts=explicit_feature_dicts,
            explicit_specs=explicit_specs,
        )
        r_loss = (y_resid - tau_hat * t_resid) ** 2
        r_loss_at_zero = y_resid**2

        predictions = pd.DataFrame(
            {
                "_oci_row_id": discovery_df["_oci_row_id"].to_numpy(),
                "outer_fold": int(outer_fold),
                "view_index": int(view_index),
                "view_name": str(view.name),
                "e_hat": e_hat,
                "m_hat": m_hat,
                "y_residual": y_resid,
                "t_residual": t_resid,
                "pseudo_target": pseudo_target,
                "tau_hat_multi_model": tau_hat,
                "r_loss": r_loss,
                "r_loss_at_zero_tau": r_loss_at_zero,
            }
        )

        metrics = self._bow_metrics(
            y=y,
            t=t,
            e_hat=e_hat,
            m_hat=m_hat,
            pseudo_target=pseudo_target,
            tau_hat=tau_hat,
            y_resid=y_resid,
            t_resid=t_resid,
            r_loss=r_loss,
            r_loss_at_zero=r_loss_at_zero,
            discovery_df=discovery_df,
        )
        importance = self._fit_feature_importance_models(
            texts=texts,
            y=y,
            t=t,
            pseudo_target=pseudo_target,
            pseudo_target_sample_weight=t_resid**2,
            view=view,
            explicit_feature_dicts=explicit_feature_dicts,
            explicit_specs=explicit_specs,
        )
        return {
            "predictions": predictions,
            "metrics": metrics,
            "importance": importance,
            "e_hat": e_hat,
            "m_hat": m_hat,
            "pseudo_target": pseudo_target,
            "t_resid": t_resid,
            "view": view,
            "view_index": int(view_index),
        }

    def _fit_ensemble_r_discovery(
        self,
        *,
        discovery_df: pd.DataFrame,
        texts: Sequence[str],
        y: np.ndarray,
        t: np.ndarray,
        outer_fold: int,
        view_results: Sequence[Dict[str, Any]],
        nuisance_results: Optional[Sequence[Dict[str, Any]]] = None,
        explicit_feature_dicts: Optional[List[Dict[str, Any]]] = None,
        explicit_specs: Optional[List[ExplicitFeatureSpec]] = None,
    ) -> Optional[Dict[str, Any]]:
        nuisance_results = [
            result for result in (nuisance_results or view_results) if result is not None
        ]
        if len(view_results) < 1 and not nuisance_results:
            return None
        if len(view_results) >= 1 and len(nuisance_results) < 2:
            return None

        e_hat = np.nanmean(
            np.vstack([np.asarray(result["e_hat"], dtype=float) for result in nuisance_results]),
            axis=0,
        )
        m_hat = np.nanmean(
            np.vstack([np.asarray(result["m_hat"], dtype=float) for result in nuisance_results]),
            axis=0,
        )
        e_clipped = np.clip(e_hat, self.nn_config.e_clip, 1.0 - self.nn_config.e_clip)
        t_resid = t - e_clipped
        y_resid = y - m_hat
        pseudo_target = y_resid / t_resid
        sample_weight = t_resid**2
        r_loss_at_zero = y_resid**2
        nuisance_source_names = [
            str(result.get("view_name") or getattr(result.get("view"), "name", "model"))
            for result in nuisance_results
        ]
        if len(nuisance_source_names) == 1:
            target_source = nuisance_source_names[0] or "single_nuisance_source"
        elif any(str(name).startswith("htr") for name in nuisance_source_names):
            target_source = "ensemble_mean_nuisance_with_htr"
        else:
            target_source = "ensemble_mean_nuisance"
        nuisance_predictions = pd.DataFrame(
            {
                "_oci_row_id": discovery_df["_oci_row_id"].to_numpy(),
                "outer_fold": int(outer_fold),
                "e_hat": e_hat,
                "m_hat": m_hat,
                "y_residual": y_resid,
                "t_residual": t_resid,
                "r_pseudo_outcome": pseudo_target,
                "pseudo_target": pseudo_target,
                "r_loss_at_zero_tau": r_loss_at_zero,
                "nuisance_fold": -1,
                "target_source": target_source,
            }
        )

        ensemble_view_results: List[Dict[str, Any]] = []
        for result in view_results:
            view = result["view"]
            view_index = int(result["view_index"])
            tau_hat = self._crossfit_pseudo_target(
                texts,
                pseudo_target,
                sample_weight,
                outer_fold,
                dina_labels=(t, y) if self.config.outcome_type == "binary" else None,
                view=view,
                view_index=view_index,
                explicit_feature_dicts=explicit_feature_dicts,
                explicit_specs=explicit_specs,
                random_seed_offset=50_000,
            )
            r_loss = (y_resid - tau_hat * t_resid) ** 2
            view_name = f"ensemble_r__{view.name}"
            predictions = pd.DataFrame(
                {
                    "_oci_row_id": discovery_df["_oci_row_id"].to_numpy(),
                    "outer_fold": int(outer_fold),
                    "view_index": view_index,
                    "view_name": view_name,
                    "e_hat": e_hat,
                    "m_hat": m_hat,
                    "y_residual": y_resid,
                    "t_residual": t_resid,
                    "pseudo_target": pseudo_target,
                    "tau_hat_multi_model": tau_hat,
                    "r_loss": r_loss,
                    "r_loss_at_zero_tau": r_loss_at_zero,
                    "target_source": target_source,
                }
            )
            metrics = self._bow_metrics(
                y=y,
                t=t,
                e_hat=e_hat,
                m_hat=m_hat,
                pseudo_target=pseudo_target,
                tau_hat=tau_hat,
                y_resid=y_resid,
                t_resid=t_resid,
                r_loss=r_loss,
                r_loss_at_zero=r_loss_at_zero,
                discovery_df=discovery_df,
            )
            importance = self._fit_feature_importance_models(
                texts=texts,
                y=y,
                t=t,
                pseudo_target=pseudo_target,
                pseudo_target_sample_weight=sample_weight,
                view=view,
                explicit_feature_dicts=explicit_feature_dicts,
                explicit_specs=explicit_specs,
            )
            ensemble_view_results.append(
                {
                    "predictions": predictions,
                    "metrics": metrics,
                    "importance": importance,
                    "pseudo_target": pseudo_target,
                    "t_resid": t_resid,
                    "view": view,
                    "view_name": view_name,
                    "view_index": view_index,
                }
            )

        metrics = _multi_view_metrics(ensemble_view_results)
        metrics["target_source"] = target_source
        metrics["n_nuisance_sources"] = int(len(nuisance_source_names))
        metrics["nuisance_sources"] = nuisance_source_names
        if len(nuisance_source_names) == 1:
            metrics["pseudo_target_construction"] = (
                f"{target_source} nuisance predictions, then " "(Y - m_hat) / (T - e_hat)"
            )
        elif target_source == "ensemble_mean_nuisance_with_htr":
            metrics["pseudo_target_construction"] = (
                "mean nuisance predictions across BoW and HTR models, then "
                "(Y - mean_m_hat) / (T - mean_e_hat)"
            )
        else:
            metrics["pseudo_target_construction"] = (
                "mean nuisance predictions across BoW views, then "
                "(Y - mean_m_hat) / (T - mean_e_hat)"
            )
        importance = _multi_view_importance(
            ensemble_view_results,
            top_n=int(self.nn_config.top_n_features),
        )
        importance["target_source"] = target_source
        importance["nuisance_sources"] = nuisance_source_names
        importance["pseudo_target_construction"] = metrics["pseudo_target_construction"]
        return {
            "view_results": ensemble_view_results,
            "metrics": metrics,
            "importance": importance,
            "pseudo_target": pseudo_target,
            "t_resid": t_resid,
            "nuisance_predictions": nuisance_predictions,
            "target_source": target_source,
        }

    def _crossfit_binary(
        self,
        texts: Sequence[str],
        labels: np.ndarray,
        label_name: str,
        outer_fold: int,
        *,
        view: BoWViewConfig,
        view_index: int,
        explicit_feature_dicts: Optional[List[Dict[str, Any]]] = None,
        explicit_specs: Optional[List[ExplicitFeatureSpec]] = None,
    ) -> np.ndarray:
        labels = labels.astype(int)
        oof = np.full(len(labels), np.nan, dtype=float)
        random_state = (
            11_000
            + 100 * outer_fold
            + 1_000 * int(view_index)
            + (1 if label_name == "outcome" else 2)
        )
        split_items = list(
            enumerate(
                _binary_split_items(
                    labels,
                    requested_folds=self.nn_config.nuisance_folds,
                    random_state=random_state,
                ),
                start=1,
            )
        )
        folds = len(split_items)
        vectorizer_params = self._vectorizer_params(view)
        model_params = self._model_params(view)

        def run_fold(fold: int, fit_pos: np.ndarray, heldout_pos: np.ndarray):
            logger.info(
                "Outer fold %s BoW view=%s %s nuisance fold %s/%s: train=%s heldout=%s",
                outer_fold,
                view.name,
                label_name,
                fold,
                folds,
                len(fit_pos),
                len(heldout_pos),
            )
            return _fit_binary_bow_fold(
                texts,
                labels,
                fit_pos,
                heldout_pos,
                vectorizer_params,
                model_params,
                explicit_feature_dicts=explicit_feature_dicts,
                explicit_specs=explicit_specs,
                random_state=17 + fold,
            )

        results = self._run_fold_tasks(run_fold, split_items)
        for heldout_pos, values in results:
            oof[heldout_pos] = values
        return np.clip(oof, self.nn_config.e_clip, 1.0 - self.nn_config.e_clip)

    def _crossfit_continuous(
        self,
        texts: Sequence[str],
        values: np.ndarray,
        label_name: str,
        outer_fold: int,
        *,
        view: BoWViewConfig,
        view_index: int,
        explicit_feature_dicts: Optional[List[Dict[str, Any]]] = None,
        explicit_specs: Optional[List[ExplicitFeatureSpec]] = None,
    ) -> np.ndarray:
        oof = np.full(len(values), np.nan, dtype=float)
        folds = _bounded_fold_count(self.nn_config.nuisance_folds, len(values))
        splitter = KFold(
            n_splits=folds,
            shuffle=True,
            random_state=12_000 + 100 * outer_fold + 1_000 * int(view_index),
        )
        split_items = list(enumerate(splitter.split(texts), start=1))
        vectorizer_params = self._vectorizer_params(view)
        model_params = self._model_params(view)

        def run_fold(fold: int, fit_pos: np.ndarray, heldout_pos: np.ndarray):
            logger.info(
                "Outer fold %s BoW view=%s %s nuisance fold %s/%s: train=%s heldout=%s",
                outer_fold,
                view.name,
                label_name,
                fold,
                folds,
                len(fit_pos),
                len(heldout_pos),
            )
            return _fit_regression_bow_fold(
                texts,
                values,
                fit_pos,
                heldout_pos,
                vectorizer_params,
                model_params,
                explicit_feature_dicts=explicit_feature_dicts,
                explicit_specs=explicit_specs,
                random_state=17 + fold,
            )

        results = self._run_fold_tasks(run_fold, split_items)
        for heldout_pos, fold_values in results:
            oof[heldout_pos] = fold_values
        return oof

    def _crossfit_pseudo_target(
        self,
        texts: Sequence[str],
        pseudo_target: np.ndarray,
        sample_weight: Optional[np.ndarray],
        outer_fold: int,
        *,
        view: BoWViewConfig,
        view_index: int,
        explicit_feature_dicts: Optional[List[Dict[str, Any]]] = None,
        explicit_specs: Optional[List[ExplicitFeatureSpec]] = None,
        random_seed_offset: int = 0,
        dina_labels=None,
    ) -> np.ndarray:
        if dina_labels is not None:
            from .stage1_dina import crossfit_effect

            result = crossfit_effect(
                texts,
                *dina_labels,
                view=view,
                folds=self.nn_config.effect_folds,
                seed=13_000 + outer_fold + 1000 * view_index + random_seed_offset,
                explicit=explicit_feature_dicts,
                specs=explicit_specs,
            )
            return result["tau"]
        oof = np.full(len(pseudo_target), np.nan, dtype=float)
        folds = _bounded_fold_count(self.nn_config.effect_folds, len(pseudo_target))
        splitter = KFold(
            n_splits=folds,
            shuffle=True,
            random_state=(13_000 + int(random_seed_offset) + outer_fold + 1_000 * int(view_index)),
        )
        split_items = list(enumerate(splitter.split(texts), start=1))
        vectorizer_params = self._vectorizer_params(view)
        model_params = self._model_params(view)

        def run_fold(fold: int, fit_pos: np.ndarray, heldout_pos: np.ndarray):
            logger.info(
                "Outer fold %s BoW view=%s pseudo-target fold %s/%s: train=%s heldout=%s",
                outer_fold,
                view.name,
                fold,
                folds,
                len(fit_pos),
                len(heldout_pos),
            )
            return _fit_regression_bow_fold(
                texts,
                pseudo_target,
                fit_pos,
                heldout_pos,
                vectorizer_params,
                model_params,
                explicit_feature_dicts=explicit_feature_dicts,
                explicit_specs=explicit_specs,
                sample_weight=sample_weight,
                random_state=17 + int(random_seed_offset) + fold,
            )

        results = self._run_fold_tasks(run_fold, split_items)
        for heldout_pos, values in results:
            oof[heldout_pos] = values
        return oof

    def _fit_feature_importance_models(
        self,
        texts: Sequence[str],
        y: np.ndarray,
        t: np.ndarray,
        pseudo_target: np.ndarray,
        pseudo_target_sample_weight: Optional[np.ndarray],
        *,
        view: BoWViewConfig,
        explicit_feature_dicts: Optional[List[Dict[str, Any]]] = None,
        explicit_specs: Optional[List[ExplicitFeatureSpec]] = None,
    ) -> Dict[str, Any]:
        vectorizer = self._make_vectorizer(view)
        x_text = vectorizer.fit_transform(texts)
        x_model, features, explicit_feature_names = _append_explicit_features_full(
            x_text,
            np.asarray(vectorizer.get_feature_names_out()),
            explicit_feature_dicts=explicit_feature_dicts,
            explicit_specs=explicit_specs,
        )

        def fit_treatment() -> np.ndarray:
            if len(np.unique(t.astype(int))) < 2:
                return np.zeros(len(features), dtype=float)
            treatment_model = self._make_classifier(view, random_state=101)
            treatment_model.fit(x_model, t.astype(int))
            return _model_feature_scores(treatment_model, len(features))

        def fit_outcome() -> np.ndarray:
            if self.config.outcome_type == "continuous":
                outcome_model = self._make_regressor(view, random_state=202)
                outcome_model.fit(x_model, y)
                return _model_feature_scores(outcome_model, len(features))
            if len(np.unique(y.astype(int))) < 2:
                return np.zeros(len(features), dtype=float)
            outcome_model = self._make_classifier(view, random_state=202)
            outcome_model.fit(x_model, y.astype(int))
            return _model_feature_scores(outcome_model, len(features))

        def fit_effect() -> np.ndarray:
            if self.config.outcome_type == "binary":
                from .stage1_dina import crossfit_nuisance, encode
                from ..models import dina

                dn, _ = crossfit_nuisance(
                    texts,
                    t,
                    y,
                    view=view,
                    folds=self.nn_config.nuisance_folds,
                    seed=303,
                    explicit=explicit_feature_dicts,
                    specs=explicit_specs,
                )
                xd, encoder = encode(
                    texts, view=view, explicit=explicit_feature_dicts, specs=explicit_specs
                )
                model = dina.fit(
                    xd,
                    y,
                    t,
                    dn["a"],
                    dn["nu"],
                    groups=encoder.groups,
                    regularization=view.ridge_alpha / len(y),
                )
                return model.coefficients
            effect_model = self._make_regressor(view, random_state=303)
            _fit_regressor(
                effect_model,
                x_model,
                pseudo_target,
                sample_weight=pseudo_target_sample_weight,
                unsupported_sample_weight_policy=(
                    view.unsupported_sample_weight_policy
                ),
            )
            return _model_feature_scores(effect_model, len(features))

        n_jobs = self._feature_importance_n_jobs()
        if n_jobs > 1:
            logger.info(
                "Multi-model BoW feature-importance parallelism: tasks=3 n_jobs=%s",
                n_jobs,
            )
            treatment_coef, outcome_coef, effect_coef = Parallel(
                n_jobs=n_jobs,
                backend="threading",
                batch_size=1,
            )(delayed(task)() for task in (fit_treatment, fit_outcome, fit_effect))
        else:
            treatment_coef = fit_treatment()
            outcome_coef = fit_outcome()
            effect_coef = fit_effect()

        top_n = int(self.nn_config.top_n_features)
        confounder_score = np.abs(treatment_coef) * np.abs(outcome_coef)
        return {
            "view_name": str(view.name),
            "view_config": _bow_view_to_dict(view),
            "effect_objective": (
                "bernoulli_dina" if self.config.outcome_type == "binary" else "squared_r_loss"
            ),
            "effect_coefficient_scale": (
                "conditional_log_odds_ratio"
                if self.config.outcome_type == "binary"
                else "outcome_difference"
            ),
            "n_features": int(len(features)),
            "n_bow_features": int(len(vectorizer.get_feature_names_out())),
            "n_prespecified_features": int(len(explicit_specs or [])),
            "n_prespecified_raw_features": int(len(explicit_feature_names)),
            "prespecified_raw_feature_names": explicit_feature_names,
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
            "treatment_positive": _top_feature_rows(
                features,
                treatment_coef,
                top_n,
                descending=True,
            ),
            "treatment_negative": _top_feature_rows(
                features,
                treatment_coef,
                top_n,
                descending=False,
            ),
            "outcome_positive": _top_feature_rows(
                features,
                outcome_coef,
                top_n,
                descending=True,
            ),
            "outcome_negative": _top_feature_rows(
                features,
                outcome_coef,
                top_n,
                descending=False,
            ),
            "pseudo_target_positive": _top_feature_rows(
                features,
                effect_coef,
                top_n,
                descending=True,
            ),
            "pseudo_target_negative": _top_feature_rows(
                features,
                effect_coef,
                top_n,
                descending=False,
            ),
        }

    def _bow_metrics(
        self,
        *,
        y: np.ndarray,
        t: np.ndarray,
        e_hat: np.ndarray,
        m_hat: np.ndarray,
        pseudo_target: np.ndarray,
        tau_hat: np.ndarray,
        y_resid: np.ndarray,
        t_resid: np.ndarray,
        r_loss: np.ndarray,
        r_loss_at_zero: np.ndarray,
        discovery_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {
            "treatment_auroc": _safe_roc_auc(t, e_hat),
            "treatment_brier": _finite_or_none(brier_score_loss(t, e_hat)),
            "treatment_log_loss": _finite_or_none(log_loss(t, e_hat)),
            "pseudo_target_mean": _finite_or_none(np.mean(pseudo_target)),
            "pseudo_target_std": _finite_or_none(np.std(pseudo_target)),
            "tau_hat_mean": _finite_or_none(np.mean(tau_hat)),
            "tau_hat_std": _finite_or_none(np.std(tau_hat)),
            "r_loss_mean": _finite_or_none(np.mean(r_loss)),
            "r_loss_at_zero_tau_mean": _finite_or_none(np.mean(r_loss_at_zero)),
            "tau_hat_pseudo_target_corr": _safe_corr(tau_hat, pseudo_target),
        }
        if self.config.outcome_type == "continuous":
            metrics["outcome_rmse"] = _finite_or_none(np.sqrt(mean_squared_error(y, m_hat)))
        else:
            metrics["outcome_auroc"] = _safe_roc_auc(y, m_hat)
            metrics["outcome_brier"] = _finite_or_none(brier_score_loss(y, m_hat))
            metrics["outcome_log_loss"] = _finite_or_none(log_loss(y, m_hat))
        zero = metrics["r_loss_at_zero_tau_mean"]
        loss = metrics["r_loss_mean"]
        if zero is not None and zero > 0 and loss is not None:
            metrics["r_loss_relative_improvement"] = float(1.0 - loss / zero)
        return metrics

    def _build_agent_context(
        self,
        *,
        outer_fold: int,
        discovery_df: pd.DataFrame,
        metrics: Dict[str, Any],
        importance: Dict[str, Any],
        embedding_evidence: Optional[Dict[str, Any]] = None,
        htr_evidence: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        enabled_methods = self._enabled_feature_discovery_methods()
        if self._evidence_digest_prompt_enabled():
            context = _build_evidence_digest_agent_context(
                outer_fold=outer_fold,
                feature_discovery_methods=enabled_methods,
                max_proposals=int(self.nn_config.candidate_proposals_per_fold),
                clinical_question=self.config.clinical_question,
                treatment_column=self.config.treatment_column,
                outcome_column=self.config.outcome_column,
                outcome_type=self.config.outcome_type,
                current_features=[_spec_to_dict(spec) for spec in self._initial_specs()],
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
                    "source": "multi_model_agentic_forest_text_models",
                    "raw_text_modeling_reused_for_agentic_stage": True,
                },
            )
            prompt_chars = len(
                json.dumps(context, separators=(",", ":"), default=_json_default)
            )
            logger.info(
                "Multi-model evidence-digest context outer_fold=%s: %.1fK JSON chars",
                outer_fold,
                prompt_chars / 1000.0,
            )
            return context
        instructions = [
            "You are generating candidate variables from empirical text "
            "evidence. The workflow will validate extraction quality, signal, "
            "parsimony, and honest causal-forest performance before retaining "
            "any variable.",
        ]
        if self._bow_discovery_enabled():
            instructions.extend(
                [
                    "Review every sparse bag-of-words model view. Each view has "
                    "its own honest nuisance predictions, R pseudo-target, "
                    "metrics, and feature-importance summaries.",
                    "Use feature_importance.phrase_consensus as a cross-view "
                    "summary, but also inspect feature_importance.views for "
                    "useful signals that appear in only one model or n-gram setting.",
                ]
            )
        else:
            instructions.append(
                "Bag-of-words modeling was disabled for this run; do not expect "
                "sparse phrase importance evidence."
            )
        if self._embedding_contrast_enabled():
            instructions.extend(
                [
                    "When embedding_contrast_evidence is present, use aligned "
                    "real-text chunks and concept scores as retrieval evidence, "
                    "not as direct vector interpretations.",
                    "For treatment, outcome, and confounder_vector embedding "
                    "contrasts, inspect both positive and negative chunk tails; "
                    "scalar confounders can be clearest in the least-aligned tail.",
                    "Treat residualized_treatment_outcome_interaction as "
                    "no-nuisance effect-modifier evidence. When available, also "
                    "use R-pseudo-target and orthogonal R-score embedding contrasts "
                    "as nuisance-model-based effect-modifier evidence.",
                    "External embedding chunks, when present, are background "
                    "retrieval evidence from another corpus and should support "
                    "clinical naming of variables rather than replace study-cohort "
                    "evidence.",
                ]
            )
        if self._htr_evidence_enabled():
            instructions.extend(
                [
                    "When htr_attention_evidence is present, use the highly "
                    "attended tokens/spans from HTR nuisance and R-stage models "
                    "as neural text evidence for variables that may explain "
                    "treatment assignment, baseline outcome risk, or heterogeneous "
                    "treatment effect.",
                    "Treat ensemble_mean_nuisance_with_htr diagnostics as the "
                    "R-loss signal built from BoW nuisance predictions plus HTR "
                    "nuisance predictions when both source families are enabled.",
                ]
            )
        instructions.extend(
            [
                "Suggest explicit pre-treatment patient-level variables, not raw text tokens.",
                "Do not invent broad clinical inventory variables unsupported by "
                "the enabled BoW, embedding, or HTR evidence in this context.",
                "Use variables predictive of both treatment and outcome as confounders.",
                "Use variables predictive of the pseudo-target as effect modifiers.",
                "Avoid near-duplicate aliases for the same extraction target; a "
                "separate alias-resolution pass may merge proposal names.",
            ]
        )
        context = {
            "prompt_version": "multi_model_agentic_forest_v1",
            "outer_fold": int(outer_fold),
            "feature_discovery_methods": enabled_methods,
            "max_proposals": int(self.nn_config.candidate_proposals_per_fold),
            "clinical_question": self.config.clinical_question,
            "estimand": {
                "treatment_column": self.config.treatment_column,
                "outcome_column": self.config.outcome_column,
                "outcome_type": self.config.outcome_type,
            },
            "instructions": instructions,
            "current_features": [_spec_to_dict(spec) for spec in self._initial_specs()],
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
        }
        if embedding_evidence:
            context["embedding_contrast_evidence"] = embedding_evidence
        if htr_evidence:
            context["htr_attention_evidence"] = htr_evidence
        compact_context = _compact_multi_model_agent_context(context)
        prompt_chars = len(
            json.dumps(compact_context, separators=(",", ":"), default=_json_default)
        )
        logger.info(
            "Multi-model agent prompt context outer_fold=%s: %.1fK JSON chars",
            outer_fold,
            prompt_chars / 1000.0,
        )
        return compact_context

    def _embedding_contrast_enabled(self) -> bool:
        embedding_config = getattr(self.nn_config, "embedding_contrast", None)
        return bool(getattr(embedding_config, "enabled", False))

    def _bow_discovery_enabled(self) -> bool:
        return bool(getattr(self.nn_config, "bow_discovery_enabled", True))

    def _htr_evidence_enabled(self) -> bool:
        return bool(getattr(self.nn_config, "htr_evidence_enabled", True))

    def _agent_context_mode(self) -> str:
        mode = str(getattr(self.nn_config, "agent_context_mode", "evidence_digest") or "")
        mode = mode.strip().lower()
        return mode if mode in {"evidence_digest", "rich_context"} else "evidence_digest"

    def _evidence_digest_prompt_enabled(self) -> bool:
        return self._agent_context_mode() == "evidence_digest"

    def _role_proposal_cap(self) -> int:
        return max(1, (int(self.nn_config.candidate_proposals_per_fold) + 1) // 2)

    def _enabled_feature_discovery_methods(self) -> List[str]:
        methods: List[str] = []
        if self._bow_discovery_enabled():
            methods.append("bow")
        if self._htr_evidence_enabled():
            methods.append("htr")
        if self._embedding_contrast_enabled():
            methods.append("embedding_contrast")
        return methods

    def _validate_required_evidence_sources(self) -> None:
        methods = self._enabled_feature_discovery_methods()
        if not methods:
            raise ValueError(
                "multi_model_agentic_forest must enable at least one feature "
                "discovery method: bow, htr, or embedding_contrast"
            )
        if not self._bow_discovery_enabled():
            reason = str(getattr(self.nn_config, "bow_discovery_disable_reason", "") or "").strip()
            logger.warning(
                "BoW discovery disabled%s",
                f": {reason}" if reason else "",
            )
        embedding_config = getattr(self.nn_config, "embedding_contrast", None)
        if not bool(getattr(embedding_config, "enabled", False)):
            reason = str(getattr(embedding_config, "disable_reason", "") or "").strip()
            if not reason:
                raise ValueError(
                    "multi_model_agentic_forest.embedding_contrast.enabled=False "
                    "requires embedding_contrast.disable_reason"
                )
            logger.warning("Embedding contrast evidence disabled: %s", reason)
        if not self._htr_evidence_enabled():
            reason = str(getattr(self.nn_config, "htr_evidence_disable_reason", "") or "").strip()
            if not reason:
                raise ValueError(
                    "multi_model_agentic_forest.htr_evidence_enabled=False "
                    "requires htr_evidence_disable_reason"
                )
            logger.warning("HTR attention/span evidence disabled: %s", reason)

    def _embedding_contrast_generator(self) -> EmbeddingContrastEvidenceGenerator:
        if self.embedding_evidence_generator is None:
            self.embedding_evidence_generator = EmbeddingContrastEvidenceGenerator(
                config=self.config,
                output_dir=self.artifact_dir,
                embedding_provider=self.embedding_provider,
            )
        return self.embedding_evidence_generator

    def _htr_provider(self) -> Any:
        if self.htr_evidence_provider is not None:
            return self.htr_evidence_provider
        if self._default_htr_evidence_provider is None:
            self._default_htr_evidence_provider = MultiModelHTREvidenceProvider(
                config=self.config,
                output_dir=self.artifact_dir,
                device=self.device,
                gpu_ids=self.gpu_ids,
                num_workers=self.num_workers,
            )
        return self._default_htr_evidence_provider

    def _fit_htr_nuisance_discovery(
        self,
        discovery_df: pd.DataFrame,
        outer_fold: int,
    ) -> Optional[Dict[str, Any]]:
        if not self._htr_evidence_enabled():
            return None
        try:
            result = self._htr_provider().fit_nuisance(discovery_df, outer_fold)
        except Exception as exc:
            raise RuntimeError("Required HTR nuisance evidence generation failed") from exc
        predictions = _align_htr_prediction_frame(
            result.get("predictions"),
            discovery_df,
            required_columns=["e_hat", "m_hat"],
            source="htr_nuisance",
        )
        attention = [dict(row) for row in result.get("attention", []) or []]
        for row in attention:
            row.setdefault("model_family", "htr")
        predictions["model_family"] = "htr"
        predictions["view_name"] = "htr_nuisance"
        predictions["target_source"] = "htr_nuisance"
        metrics = _htr_nuisance_metrics(
            discovery_df=discovery_df,
            predictions=predictions,
            treatment_column=self.config.treatment_column,
            outcome_column=self.config.outcome_column,
            outcome_type=self.config.outcome_type,
        )
        return {
            "model_family": "htr",
            "view_name": "htr_nuisance",
            "predictions": predictions,
            "attention": attention,
            "metrics": metrics,
            "e_hat": predictions["e_hat"].to_numpy(dtype=float),
            "m_hat": predictions["m_hat"].to_numpy(dtype=float),
        }

    def _fit_htr_effect_discovery(
        self,
        discovery_df: pd.DataFrame,
        outer_fold: int,
        nuisance_predictions: pd.DataFrame,
    ) -> Optional[Dict[str, Any]]:
        if not self._htr_evidence_enabled():
            return None
        try:
            result = self._htr_provider().fit_effect(
                discovery_df,
                nuisance_predictions,
                outer_fold,
            )
        except Exception as exc:
            raise RuntimeError("Required HTR effect evidence generation failed") from exc
        predictions = _align_htr_prediction_frame(
            result.get("predictions"),
            discovery_df,
            required_columns=["tau_hat_r_stage"],
            source="htr_effect",
        )
        attention = [dict(row) for row in result.get("attention", []) or []]
        for row in attention:
            row.setdefault("model_family", "htr")
        predictions["model_family"] = "htr"
        predictions["view_name"] = "htr_effect"
        predictions["target_source"] = "ensemble_mean_nuisance_with_htr"
        metrics = _htr_effect_metrics(predictions)
        return {
            "model_family": "htr",
            "view_name": "htr_effect",
            "predictions": predictions,
            "attention": attention,
            "metrics": metrics,
            "tau_hat": predictions["tau_hat_r_stage"].to_numpy(dtype=float),
        }

    def _build_embedding_contrast_evidence(
        self,
        *,
        discovery_df: pd.DataFrame,
        y: np.ndarray,
        t: np.ndarray,
        pseudo_target: Any,
        t_resid: Any,
        pseudo_target_names: Optional[Sequence[str]] = None,
        importance: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not self._embedding_contrast_enabled():
            return {}
        try:
            generator = self._embedding_contrast_generator()
            generator.prepare(self.dataset)
            ordered_fit_rows = tuple(
                discovery_df["_oci_row_id"].astype(int).tolist()
            )
            generator.bind_cluster_fit_context(
                ordered_fit_row_ids=ordered_fit_rows,
                canonical_group_seed=derive_discovery_seed(
                    _required_applied_seed(self.config),
                    ordered_fit_rows,
                ),
            )
            dina_nuisance = None
            if self.config.outcome_type == "binary":
                from .stage1_dina import crossfit_nuisance

                dina_nuisance, _ = crossfit_nuisance(
                    discovery_df[self.config.text_column].astype(str).tolist(),
                    t,
                    y,
                    folds=self.nn_config.nuisance_folds,
                    seed=49_001,
                )
            return generator.build_evidence(
                dina_nuisance=dina_nuisance,
                discovery_df=discovery_df,
                y=y,
                t=t,
                pseudo_target=pseudo_target,
                t_resid=t_resid,
                pseudo_target_names=(
                    pseudo_target_names
                    if pseudo_target_names is not None
                    else [view.name for view in self.nn_config.bow_views]
                ),
                importance=importance,
            )
        except Exception as exc:
            raise RuntimeError("Required embedding contrast evidence generation failed") from exc

    def _artifact_agent_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if self.search_config.save_agent_context:
            return context
        if (
            "embedding_contrast_evidence" not in context
            and "htr_attention_evidence" not in context
            and "evidence_digest" not in context
        ):
            return context
        artifact_context = dict(context)
        if "embedding_contrast_evidence" in context:
            artifact_context["embedding_contrast_evidence"] = redact_embedding_contrast_evidence(
                context["embedding_contrast_evidence"]
            )
        if "htr_attention_evidence" in context:
            artifact_context["htr_attention_evidence"] = _redact_htr_attention_evidence(
                context["htr_attention_evidence"]
            )
        if "evidence_digest" in context:
            artifact_context["evidence_digest"] = _redact_evidence_digest(
                context["evidence_digest"]
            )
        return artifact_context

    def _propose_selected_specs(
        self,
        *,
        outer_fold: int,
        discovery_df: pd.DataFrame,
        bow_context: Dict[str, Any],
    ) -> List[ExplicitFeatureSpec]:
        if bool(getattr(self.nn_config, "candidate_consistency_enabled", True)):
            return self._propose_selected_specs_with_consistency(
                outer_fold=outer_fold,
                discovery_df=discovery_df,
                bow_context=bow_context,
            )
        return self._propose_selected_specs_without_consistency(
            outer_fold=outer_fold,
            discovery_df=discovery_df,
            bow_context=bow_context,
        )

    def _propose_selected_specs_without_consistency(
        self,
        *,
        outer_fold: int,
        discovery_df: pd.DataFrame,
        bow_context: Dict[str, Any],
    ) -> List[ExplicitFeatureSpec]:
        cached = self._load_selected_specs_cache(
            outer_fold=outer_fold,
            consistency_enabled=False,
        )
        if cached is not None:
            selected_specs, row = cached
            self._remember_alias_reference_specs(selected_specs)
            self.agent_rows.append(row)
            return selected_specs

        bundle = self._propose_candidate_bundle(
            outer_fold=outer_fold,
            scope="full_outer_train",
            bow_context=bow_context,
            n_rows=len(discovery_df),
        )
        proposals = list(bundle.get("valid_proposals", []) or [])
        proposals, alias_resolution = self._resolve_proposal_aliases(
            outer_fold=outer_fold,
            proposals=proposals,
        )
        selected_specs = _dedupe_specs(
            [
                *self._initial_specs(),
                *[
                    ExplicitFeatureSpec(
                        name=proposal.name,
                        type=proposal.type or "continuous",
                        categories=proposal.categories,
                        roles=proposal.roles,
                        description=proposal.description,
                    )
                    for proposal in proposals
                    if proposal.action == "add"
                ],
            ]
        )
        selected_specs, value_harmonization = self._harmonize_value_contracts(
            outer_fold=outer_fold,
            selected_specs=selected_specs,
        )
        self._remember_alias_reference_specs(selected_specs)
        row: Dict[str, Any] = {
            "outer_fold": int(outer_fold),
            "prompt_context_mode": self._agent_context_mode(),
            "raw_proposals": bundle.get("raw_proposals"),
            "alias_resolution": alias_resolution,
            "value_harmonization": value_harmonization,
            "valid_proposals": [
                {
                    "action": proposal.action,
                    "name": proposal.name,
                    "type": proposal.type,
                    "categories": proposal.categories,
                    "roles": proposal.roles,
                    "description": proposal.description,
                    "rationale": proposal.rationale,
                    "expected_signal": proposal.expected_signal,
                }
                for proposal in proposals
            ],
            "rejected_proposals": bundle.get("rejected_proposals", []),
            "selected_features": [_spec_to_dict(spec) for spec in selected_specs],
        }
        for key in ["raw_proposals_by_role", "role_candidate_caps", "prompt_versions"]:
            if key in bundle:
                row[key] = bundle[key]
        if bundle.get("concept_inventory") is not None:
            row["concept_inventory"] = bundle.get("concept_inventory")
        if self.search_config.save_agent_context:
            for key in ["context", "contexts_by_role"]:
                if key in bundle:
                    row[key] = bundle[key]
        if self.search_config.save_agent_raw_output:
            for key in ["agent_raw_output", "agent_raw_output_by_role"]:
                if key in bundle:
                    row[key] = bundle[key]
        self.agent_rows.append(row)
        self._write_selected_specs_cache(
            outer_fold=outer_fold,
            consistency_enabled=False,
            selected_specs=selected_specs,
            row=row,
        )
        return selected_specs

    def _propose_selected_specs_with_consistency(
        self,
        *,
        outer_fold: int,
        discovery_df: pd.DataFrame,
        bow_context: Dict[str, Any],
    ) -> List[ExplicitFeatureSpec]:
        cached = self._load_selected_specs_cache(
            outer_fold=outer_fold,
            consistency_enabled=True,
        )
        if cached is not None:
            selected_specs, row = cached
            self._remember_alias_reference_specs(selected_specs)
            self.agent_rows.append(row)
            return selected_specs

        full_bundle = self._propose_candidate_bundle(
            outer_fold=outer_fold,
            scope="full_outer_train",
            bow_context={
                **bow_context,
                "consistency_scope": "full_outer_train",
            },
            n_rows=len(discovery_df),
        )
        bundles = [
            full_bundle,
            *self._inner_consistency_candidate_bundles(
                outer_fold=outer_fold,
                discovery_df=discovery_df,
            ),
        ]
        all_proposals = [
            proposal
            for bundle in bundles
            for proposal in bundle.get("valid_proposals", [])
            if proposal.action == "add"
        ]
        if not all_proposals:
            selected_specs = self._initial_specs()
            selected_specs, value_harmonization = self._harmonize_value_contracts(
                outer_fold=outer_fold,
                selected_specs=selected_specs,
            )
            row = {
                "outer_fold": int(outer_fold),
                "consistency_enabled": True,
                "proposal_bundles": [_proposal_bundle_artifact(bundle) for bundle in bundles],
                "selected_features": [_spec_to_dict(spec) for spec in selected_specs],
                "value_harmonization": value_harmonization,
                "skipped": "no_valid_consistency_candidates",
            }
            self.agent_rows.append(row)
            self._write_selected_specs_cache(
                outer_fold=outer_fold,
                consistency_enabled=True,
                selected_specs=selected_specs,
                row=row,
            )
            return selected_specs

        alias_input = _merge_duplicate_proposals(all_proposals)
        alias_resolved, alias_resolution = self._resolve_proposal_aliases(
            outer_fold=outer_fold,
            proposals=alias_input,
        )
        alias_map = {
            item["from"]: item["to"]
            for item in alias_resolution.get("applied_aliases", [])
            if item.get("from") and item.get("to")
        }
        canonical_proposals = {
            proposal.name: proposal for proposal in alias_resolved if proposal.action == "add"
        }
        candidate_summaries, threshold, inner_fold_count = (
            self._build_consistency_candidate_summaries(
                bundles=bundles,
                alias_map=alias_map,
                canonical_proposals=canonical_proposals,
            )
        )
        consistency_context = self._build_consistency_context(
            outer_fold=outer_fold,
            candidate_summaries=candidate_summaries,
            threshold=threshold,
            inner_fold_count=inner_fold_count,
        )
        consistency_proposals, consistency_selection = self._select_consistent_proposals(
            context=consistency_context,
            candidate_summaries=candidate_summaries,
            canonical_proposals=canonical_proposals,
        )
        selected_specs = self._selected_specs_from_proposals(consistency_proposals)
        selected_specs, value_harmonization = self._harmonize_value_contracts(
            outer_fold=outer_fold,
            selected_specs=selected_specs,
        )
        self._remember_alias_reference_specs(selected_specs)

        row: Dict[str, Any] = {
            "outer_fold": int(outer_fold),
            "consistency_enabled": True,
            "proposal_bundles": [_proposal_bundle_artifact(bundle) for bundle in bundles],
            "alias_resolution": alias_resolution,
            "consistency": {
                "inner_fold_count": int(inner_fold_count),
                "min_support_folds": int(threshold),
                "candidate_summaries": candidate_summaries,
                "selection": consistency_selection,
            },
            "value_harmonization": value_harmonization,
            "selected_features": [_spec_to_dict(spec) for spec in selected_specs],
        }
        if self.search_config.save_agent_context:
            row["consistency_context"] = consistency_context
        self.agent_rows.append(row)
        self._write_selected_specs_cache(
            outer_fold=outer_fold,
            consistency_enabled=True,
            selected_specs=selected_specs,
            row=row,
        )
        return selected_specs

    def _propose_candidate_bundle(
        self,
        *,
        outer_fold: int,
        scope: str,
        bow_context: Dict[str, Any],
        n_rows: int,
        inner_fold: Optional[int] = None,
        heldout_rows: Optional[int] = None,
    ) -> Dict[str, Any]:
        cached = self._load_proposal_bundle_cache(
            outer_fold=outer_fold,
            scope=scope,
            inner_fold=inner_fold,
        )
        if cached is not None:
            return cached

        if self._evidence_digest_prompt_enabled():
            return self._propose_digest_candidate_bundle(
                outer_fold=outer_fold,
                scope=scope,
                bow_context=bow_context,
                n_rows=n_rows,
                inner_fold=inner_fold,
                heldout_rows=heldout_rows,
            )

        concept_inventory = self._run_concept_inventory(
            outer_fold=outer_fold,
            scope=scope,
            inner_fold=inner_fold,
            bow_context=bow_context,
            n_rows=n_rows,
            heldout_rows=heldout_rows,
        )
        proposal_context = self._proposal_context_with_concept_inventory(
            bow_context,
            concept_inventory,
        )
        raw_proposals = self.proposal_agent.propose(proposal_context)
        proposal_agent_trace = _get_agent_response_trace(self.proposal_agent)
        proposals, rejected = validate_agentic_proposals(
            raw_proposals,
            current_specs=self._initial_specs(),
            search_config=self.search_config,
            allow_removals=False,
            max_additions=self.nn_config.candidate_proposals_per_fold,
        )
        bundle: Dict[str, Any] = {
            "outer_fold": int(outer_fold),
            "scope": scope,
            "inner_fold": inner_fold,
            "n_rows": int(n_rows),
            "heldout_rows": None if heldout_rows is None else int(heldout_rows),
            "prompt_context_mode": self._agent_context_mode(),
            "prompt_versions": [str(proposal_context.get("prompt_version") or "")],
            "raw_proposals": raw_proposals,
            "valid_proposals": proposals,
            "rejected_proposals": rejected,
        }
        if concept_inventory is not None:
            bundle["concept_inventory"] = _concept_inventory_artifact(concept_inventory)
        if self.search_config.save_agent_context:
            bundle["context"] = proposal_context
        if self.search_config.save_agent_raw_output:
            bundle["agent_raw_output"] = proposal_agent_trace
        self._write_proposal_bundle_cache(bundle)
        return bundle

    def _propose_digest_candidate_bundle(
        self,
        *,
        outer_fold: int,
        scope: str,
        bow_context: Dict[str, Any],
        n_rows: int,
        inner_fold: Optional[int] = None,
        heldout_rows: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not isinstance(bow_context.get("evidence_digest"), dict):
            bow_context = _evidence_digest_context_from_rich_context(bow_context)
        role_cap = self._role_proposal_cap()
        raw_by_role: Dict[str, Any] = {}
        context_by_role: Dict[str, Dict[str, Any]] = {}
        trace_by_role: Dict[str, Any] = {}
        rejected: List[Dict[str, Any]] = []
        valid: List[AgenticFeatureProposal] = []

        for role in ["confounder", "effect_modifier"]:
            proposal_context = _evidence_digest_role_context(
                bow_context,
                role=role,
                max_proposals=role_cap,
            )
            raw_proposals = self.proposal_agent.propose(proposal_context)
            raw_by_role[role] = raw_proposals
            if self.search_config.save_agent_context:
                context_by_role[role] = proposal_context
            if self.search_config.save_agent_raw_output:
                trace_by_role[role] = _get_agent_response_trace(self.proposal_agent)
            role_raw = _role_forced_raw_proposals(raw_proposals, role)
            role_proposals, role_rejected = validate_agentic_proposals(
                role_raw,
                current_specs=self._initial_specs(),
                search_config=self.search_config,
                allow_removals=False,
                max_additions=role_cap,
            )
            valid.extend(role_proposals)
            for item in role_rejected:
                row = dict(item)
                row["target_role"] = role
                rejected.append(row)

        bundle: Dict[str, Any] = {
            "outer_fold": int(outer_fold),
            "scope": scope,
            "inner_fold": inner_fold,
            "n_rows": int(n_rows),
            "heldout_rows": None if heldout_rows is None else int(heldout_rows),
            "prompt_context_mode": self._agent_context_mode(),
            "prompt_versions": [_EVIDENCE_DIGEST_ROLE_PROMPT_VERSION],
            "role_candidate_caps": {
                "confounder": int(role_cap),
                "effect_modifier": int(role_cap),
            },
            "raw_proposals_by_role": raw_by_role,
            "raw_proposals": [
                proposal
                for role in ["confounder", "effect_modifier"]
                for proposal in _role_forced_raw_proposals(raw_by_role.get(role), role)
            ],
            "valid_proposals": _merge_duplicate_proposals(valid),
            "rejected_proposals": rejected,
        }
        if self.search_config.save_agent_context:
            bundle["contexts_by_role"] = context_by_role
        if self.search_config.save_agent_raw_output:
            bundle["agent_raw_output_by_role"] = trace_by_role
        self._write_proposal_bundle_cache(bundle)
        return bundle

    def _outer_fold_artifact_dir(self, outer_fold: int) -> Path:
        return self.artifact_dir / f"outer_fold_{int(outer_fold):03d}"

    def _selected_specs_cache_path(self, outer_fold: int) -> Path:
        return self._outer_fold_artifact_dir(outer_fold) / "selected_specs_cache.json"

    def _proposal_bundle_cache_path(
        self,
        *,
        outer_fold: int,
        scope: str,
        inner_fold: Optional[int],
    ) -> Path:
        if inner_fold is None:
            stem = str(scope)
        else:
            stem = f"{scope}_inner_{int(inner_fold):03d}"
        return (
            self._outer_fold_artifact_dir(outer_fold)
            / "proposal_bundles"
            / f"{stem}.json"
        )

    def _concept_inventory_cache_path(
        self,
        *,
        outer_fold: int,
        scope: str,
        inner_fold: Optional[int],
    ) -> Path:
        if inner_fold is None:
            stem = str(scope)
        else:
            stem = f"{scope}_inner_{int(inner_fold):03d}"
        return (
            self._outer_fold_artifact_dir(outer_fold)
            / "concept_inventories"
            / f"{stem}.json"
        )

    def _concept_inventory_enabled(self) -> bool:
        return bool(getattr(self.nn_config, "concept_inventory_enabled", True)) and not (
            self._evidence_digest_prompt_enabled()
        )

    def _build_concept_cluster_label_context(
        self,
        *,
        outer_fold: int,
        scope: str,
        bow_context: Dict[str, Any],
        cluster_payload: Dict[str, Any],
        n_rows: int,
        inner_fold: Optional[int],
        heldout_rows: Optional[int],
    ) -> Dict[str, Any]:
        methods = (
            bow_context.get("feature_discovery_methods")
            or self._enabled_feature_discovery_methods()
        )
        if isinstance(methods, str):
            method_list = [methods]
        else:
            method_list = list(methods)
        context: Dict[str, Any] = {
            "prompt_version": _CONCEPT_CLUSTER_LABEL_PROMPT_VERSION,
            "schema_version": _CONCEPT_INVENTORY_SCHEMA_VERSION,
            "outer_fold": int(outer_fold),
            "scope": str(scope),
            "inner_fold": None if inner_fold is None else int(inner_fold),
            "n_rows": int(n_rows),
            "heldout_rows": None if heldout_rows is None else int(heldout_rows),
            "feature_discovery_methods": method_list,
            "max_concepts": int(getattr(self.nn_config, "concept_inventory_max_concepts", 60)),
            "cluster_generation": cluster_payload.get("generation", {}),
            "labeling_mode": cluster_payload.get("labeling_mode", "single_cluster"),
            "clusters": cluster_payload.get("agent_clusters", []),
            "instructions": [
                "Label and merge only the supplied evidence clusters.",
                "Each cluster was generated without clinical dictionaries from BoW phrases, embedding-retrieved chunks, and HTR snippets.",
                "A single cluster may contain several distinct patient-level fields; emit separate concepts for each supported field and reuse the same cluster_id across those concepts.",
                "For mixed clinical panels such as CBCs, CMPs, molecular panels, vitals, demographics, or pathology/IHC groups, consider the individual represented components rather than only the panel-level label.",
                "Name patient-level concepts represented by one or more clusters; reject boilerplate, document structure, and non-patient-level clusters.",
                "Do not decide confounder/effect-modifier roles here.",
                "Do not invent concepts that are not supported by the supplied clusters.",
            ],
            "response_contract": {
                "concepts": [
                    {
                        "name": "snake_case_concept_name",
                        "label": "short label",
                        "value_kind": "binary|categorical|continuous|ordinal|text|unknown",
                        "source_families": ["bow", "embedding_contrast", "htr"],
                        "source_overlap": 2,
                        "supporting_phrases": ["source phrase"],
                        "example_values_or_phrases": ["example value or phrase"],
                        "extractability": "high|medium|low",
                        "cluster_ids": ["tfidf_svd_001", "embedding_004"],
                        "notes": "brief source-grounded explanation",
                    }
                ],
                "rejected_clusters": [
                    {
                        "cluster_id": "cluster_id",
                        "reason": "why this cluster is boilerplate, too broad, or not patient-level",
                    }
                ],
            },
        }
        return context

    def _get_concept_cluster_embedding_encoder(self) -> Tuple[Optional[Any], Optional[str]]:
        embedding_config = getattr(self.nn_config, "embedding_contrast", None)
        model_name = str(getattr(embedding_config, "model_name", "") or "").strip()
        if not model_name:
            return None, "embedding_model_name_missing"
        device = getattr(embedding_config, "device", None)
        cache_dir = getattr(embedding_config, "cache_dir", None)
        encoder_key = (
            model_name,
            "" if device is None else str(device),
            "" if cache_dir is None else str(cache_dir),
        )
        if (
            self._concept_cluster_embedding_encoder is not None
            and self._concept_cluster_embedding_encoder_key == encoder_key
        ):
            return self._concept_cluster_embedding_encoder, None

        self._release_concept_cluster_embedding_encoder(empty_cache=False)
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            return None, f"sentence_transformers_unavailable:{exc.__class__.__name__}: {exc}"

        try:
            logger.info(
                "Loading concept-cluster embedding encoder model=%s device=%s",
                model_name,
                device,
            )
            encoder = SentenceTransformer(
                model_name,
                device=device,
                cache_folder=cache_dir,
            )
        except Exception as exc:
            return None, f"sentence_transformer_load_failed:{exc.__class__.__name__}: {exc}"

        self._concept_cluster_embedding_encoder = encoder
        self._concept_cluster_embedding_encoder_key = encoder_key
        return encoder, None

    def _release_concept_cluster_embedding_encoder(self, *, empty_cache: bool = True) -> None:
        encoder = self._concept_cluster_embedding_encoder
        self._concept_cluster_embedding_encoder = None
        self._concept_cluster_embedding_encoder_key = None
        if encoder is None:
            if empty_cache and torch.cuda.is_available():
                torch.cuda.empty_cache()
            return
        try:
            if hasattr(encoder, "to"):
                encoder.to("cpu")
        except Exception:
            logger.debug("Ignoring error while moving concept-cluster encoder to CPU", exc_info=True)
        del encoder
        gc.collect()
        if empty_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _encode_concept_cluster_texts(self, texts: Sequence[str]) -> Tuple[Optional[np.ndarray], Optional[str]]:
        clean_texts = [str(text or "") for text in texts]
        if not clean_texts:
            return None, "no_texts"

        provider = self.embedding_provider
        try:
            if provider is not None and hasattr(provider, "encode_chunks"):
                matrix = provider.encode_chunks(clean_texts)
                return _coerce_concept_embedding_matrix(matrix, len(clean_texts)), None
            if provider is not None and hasattr(provider, "encode"):
                matrix = provider.encode(clean_texts)
                return _coerce_concept_embedding_matrix(matrix, len(clean_texts)), None
            if provider is not None and hasattr(provider, "encode_texts"):
                matrix = provider.encode_texts(clean_texts)
                return _coerce_concept_embedding_matrix(matrix, len(clean_texts)), None
        except Exception as exc:
            return None, f"embedding_provider_failed:{exc.__class__.__name__}: {exc}"

        if not self._embedding_contrast_enabled():
            return None, "embedding_contrast_disabled"

        embedding_config = getattr(self.nn_config, "embedding_contrast", None)
        encoder, load_error = self._get_concept_cluster_embedding_encoder()
        if load_error:
            return None, load_error
        try:
            matrix = encoder.encode(
                clean_texts,
                batch_size=int(getattr(embedding_config, "batch_size", 16)),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return _coerce_concept_embedding_matrix(matrix, len(clean_texts)), None
        except Exception as exc:
            self._release_concept_cluster_embedding_encoder()
            return None, f"sentence_transformer_encode_failed:{exc.__class__.__name__}: {exc}"

    def _load_concept_inventory_cache(
        self,
        *,
        outer_fold: int,
        scope: str,
        inner_fold: Optional[int],
    ) -> Optional[Dict[str, Any]]:
        if not self.resume:
            return None
        path = self._concept_inventory_cache_path(
            outer_fold=outer_fold,
            scope=scope,
            inner_fold=inner_fold,
        )
        if not path.exists():
            return None
        try:
            payload = _read_json(path)
            if not isinstance(payload, dict):
                return None
            if int(payload.get("outer_fold", -1)) != int(outer_fold):
                return None
            if str(payload.get("scope")) != str(scope):
                return None
            if payload.get("schema_version") != _CONCEPT_INVENTORY_SCHEMA_VERSION:
                logger.info(
                    "Ignoring cached Stage 2 concept inventory with legacy schema "
                    "outer_fold=%s scope=%s inner_fold=%s path=%s schema=%s",
                    outer_fold,
                    scope,
                    inner_fold,
                    path,
                    payload.get("schema_version"),
                )
                return None
            payload_inner = payload.get("inner_fold")
            if (None if payload_inner is None else int(payload_inner)) != (
                None if inner_fold is None else int(inner_fold)
            ):
                return None
            if not isinstance(payload.get("concepts"), list):
                return None
            payload["resumed_from_cache"] = str(path)
            logger.info(
                "Reusing cached Stage 2 concept inventory outer_fold=%s scope=%s "
                "inner_fold=%s path=%s concepts=%s",
                outer_fold,
                scope,
                inner_fold,
                path,
                len(payload.get("concepts") or []),
            )
            return payload
        except Exception as exc:
            logger.warning("Ignoring unreadable concept inventory cache %s: %s", path, exc)
            return None

    def _write_concept_inventory_cache(self, inventory: Dict[str, Any]) -> None:
        path = self._concept_inventory_cache_path(
            outer_fold=int(inventory["outer_fold"]),
            scope=str(inventory["scope"]),
            inner_fold=inventory.get("inner_fold"),
        )
        _write_json(path, _concept_inventory_artifact(inventory))

    def _run_concept_inventory(
        self,
        *,
        outer_fold: int,
        scope: str,
        bow_context: Dict[str, Any],
        n_rows: int,
        inner_fold: Optional[int] = None,
        heldout_rows: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self._concept_inventory_enabled():
            return None
        cached = self._load_concept_inventory_cache(
            outer_fold=outer_fold,
            scope=scope,
            inner_fold=inner_fold,
        )
        if cached is not None:
            return cached
        max_concepts = int(getattr(self.nn_config, "concept_inventory_max_concepts", 60))
        cluster_payload = _build_clustered_concept_inventory_payload(
            bow_context=bow_context,
            outer_fold=outer_fold,
            max_concepts=max_concepts,
            embedding_encoder=self._encode_concept_cluster_texts,
        )
        concepts: List[Dict[str, Any]] = []
        rejected_clusters: List[Dict[str, Any]] = []
        cluster_responses: List[Dict[str, Any]] = []
        inventory_contexts: List[Dict[str, Any]] = []
        inventory_agent_traces: List[Dict[str, Any]] = []
        labeling_errors: List[Dict[str, str]] = []
        fallback_used = False
        agent_clusters = list(cluster_payload.get("agent_clusters", []) or [])
        if agent_clusters:
            for cluster in agent_clusters:
                cluster_id = str(cluster.get("cluster_id") or "")
                cluster_context = self._build_concept_cluster_label_context(
                    outer_fold=outer_fold,
                    scope=scope,
                    bow_context=bow_context,
                    cluster_payload={
                        "generation": cluster_payload.get("generation", {}),
                        "labeling_mode": "single_cluster",
                        "agent_clusters": [cluster],
                    },
                    n_rows=n_rows,
                    inner_fold=inner_fold,
                    heldout_rows=heldout_rows,
                )
                if self.search_config.save_agent_context:
                    inventory_contexts.append(cluster_context)
                response: Dict[str, Any] = {}
                try:
                    raw_response = self.proposal_agent.propose(cluster_context)
                    agent_trace = _get_agent_response_trace(self.proposal_agent)
                    if self.search_config.save_agent_raw_output and agent_trace is not None:
                        inventory_agent_traces.append(
                            {"cluster_id": cluster_id, "agent_raw_output": agent_trace}
                        )
                    response = raw_response if isinstance(raw_response, dict) else {}
                except Exception as exc:
                    labeling_errors.append(
                        {
                            "cluster_id": cluster_id,
                            "error": f"{exc.__class__.__name__}: {exc}",
                        }
                    )
                    logger.warning(
                        "Cluster concept labeling failed; using deterministic "
                        "cluster label outer_fold=%s scope=%s inner_fold=%s cluster_id=%s",
                        outer_fold,
                        scope,
                        inner_fold,
                        cluster_id,
                        exc_info=True,
                    )

                cluster_concepts = response.get("concepts", []) if response else []
                cluster_concepts = cluster_concepts if isinstance(cluster_concepts, list) else []
                cluster_rejected = response.get("rejected_clusters", []) if response else []
                cluster_rejected = cluster_rejected if isinstance(cluster_rejected, list) else []
                cluster_fallback_used = False
                if not cluster_concepts and not cluster_rejected:
                    fallback_used = True
                    cluster_fallback_used = True
                    cluster_concepts = _fallback_concepts_from_clusters(
                        [cluster],
                        max_concepts=max_concepts,
                    )
                concepts.extend(item for item in cluster_concepts if isinstance(item, dict))
                rejected_clusters.extend(
                    item for item in cluster_rejected if isinstance(item, dict)
                )
                cluster_responses.append(
                    {
                        "cluster_id": cluster_id,
                        "response": response,
                        "fallback_used": bool(cluster_fallback_used),
                    }
                )
            concepts = _merge_concept_inventory_concepts(
                concepts,
                max_concepts=max_concepts,
            )
        else:
            labeling_errors.append(
                {"cluster_id": "", "error": "no_clusters_to_label"}
            )
        if not concepts:
            fallback_used = True
            concepts = _fallback_concepts_from_clusters(
                agent_clusters,
                max_concepts=max_concepts,
            )
        response = {
            "mode": "per_cluster",
            "cluster_responses": cluster_responses,
            "concepts": concepts,
            "rejected_clusters": rejected_clusters,
        }
        inventory: Dict[str, Any] = {
            "schema_version": _CONCEPT_INVENTORY_SCHEMA_VERSION,
            "outer_fold": int(outer_fold),
            "scope": str(scope),
            "inner_fold": None if inner_fold is None else int(inner_fold),
            "n_rows": int(n_rows),
            "heldout_rows": None if heldout_rows is None else int(heldout_rows),
            "generation": cluster_payload.get("generation", {}),
            "clusters": cluster_payload.get("agent_clusters", []),
            "response": response,
            "concepts": concepts,
            "fallback_used": bool(fallback_used),
        }
        if labeling_errors:
            inventory["labeling_errors"] = labeling_errors
        if self.search_config.save_agent_context:
            inventory["context"] = {
                "mode": "per_cluster",
                "cluster_contexts": inventory_contexts,
            }
        if self.search_config.save_agent_raw_output and inventory_agent_traces:
            inventory["agent_raw_output"] = inventory_agent_traces
        self._write_concept_inventory_cache(inventory)
        logger.info(
            "Generated clustered Stage 2 concept inventory outer_fold=%s scope=%s "
            "inner_fold=%s concepts=%s clusters=%s fallback=%s",
            outer_fold,
            scope,
            inner_fold,
            len(inventory["concepts"]),
            len(inventory.get("clusters") or []),
            fallback_used,
        )
        return inventory

    def _proposal_context_with_concept_inventory(
        self,
        bow_context: Dict[str, Any],
        concept_inventory: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        visible = _agent_visible_concept_inventory(
            concept_inventory,
            max_concepts=int(getattr(self.nn_config, "concept_inventory_max_concepts", 60)),
        )
        if visible is None:
            return bow_context
        return {**bow_context, "concept_inventory": visible}

    def _load_selected_specs_cache(
        self,
        *,
        outer_fold: int,
        consistency_enabled: bool,
    ) -> Optional[Tuple[List[ExplicitFeatureSpec], Dict[str, Any]]]:
        if not self.resume:
            return None
        path = self._selected_specs_cache_path(outer_fold)
        if not path.exists():
            return None
        try:
            payload = _read_json(path)
            if not isinstance(payload, dict):
                return None
            if bool(payload.get("consistency_enabled")) != bool(consistency_enabled):
                return None
            if str(payload.get("prompt_context_mode") or "") != self._agent_context_mode():
                logger.info(
                    "Ignoring cached Stage 2 selected specs with stale prompt context "
                    "mode outer_fold=%s path=%s cached=%s current=%s",
                    outer_fold,
                    path,
                    payload.get("prompt_context_mode"),
                    self._agent_context_mode(),
                )
                return None
            selected_specs = [
                _feature_spec_from_dict(item)
                for item in payload.get("selected_features", [])
                if isinstance(item, dict)
            ]
            row = dict(payload.get("agent_row") or {})
            if self._concept_inventory_enabled() and not _agent_row_has_concept_inventory(
                row,
                consistency_enabled=consistency_enabled,
            ):
                logger.info(
                    "Ignoring cached Stage 2 selected specs without concept "
                    "inventory outer_fold=%s path=%s",
                    outer_fold,
                    path,
                )
                return None
            row.setdefault("outer_fold", int(outer_fold))
            row["resumed_from_cache"] = str(path)
            logger.info(
                "Reusing cached Stage 2 selected specs outer_fold=%s path=%s",
                outer_fold,
                path,
            )
            return selected_specs, row
        except Exception as exc:
            logger.warning("Ignoring unreadable selected-spec cache %s: %s", path, exc)
            return None

    def _write_selected_specs_cache(
        self,
        *,
        outer_fold: int,
        consistency_enabled: bool,
        selected_specs: Sequence[ExplicitFeatureSpec],
        row: Dict[str, Any],
    ) -> None:
        payload = {
            "outer_fold": int(outer_fold),
            "consistency_enabled": bool(consistency_enabled),
            "prompt_context_mode": self._agent_context_mode(),
            "selected_features": [_spec_to_dict(spec) for spec in selected_specs],
            "agent_row": row,
        }
        _write_json(self._selected_specs_cache_path(outer_fold), payload)

    def _load_proposal_bundle_cache(
        self,
        *,
        outer_fold: int,
        scope: str,
        inner_fold: Optional[int],
    ) -> Optional[Dict[str, Any]]:
        if not self.resume:
            return None
        path = self._proposal_bundle_cache_path(
            outer_fold=outer_fold,
            scope=scope,
            inner_fold=inner_fold,
        )
        if not path.exists():
            return None
        try:
            payload = _read_json(path)
            if not isinstance(payload, dict):
                return None
            if int(payload.get("outer_fold", -1)) != int(outer_fold):
                return None
            if str(payload.get("scope")) != str(scope):
                return None
            if str(payload.get("prompt_context_mode") or "") != self._agent_context_mode():
                logger.info(
                    "Ignoring cached Stage 2 proposal bundle with stale prompt context "
                    "mode outer_fold=%s scope=%s inner_fold=%s path=%s cached=%s current=%s",
                    outer_fold,
                    scope,
                    inner_fold,
                    path,
                    payload.get("prompt_context_mode"),
                    self._agent_context_mode(),
                )
                return None
            payload_inner = payload.get("inner_fold")
            if (None if payload_inner is None else int(payload_inner)) != (
                None if inner_fold is None else int(inner_fold)
            ):
                return None
            if self._concept_inventory_enabled() and not isinstance(
                payload.get("concept_inventory"),
                dict,
            ):
                logger.info(
                    "Ignoring cached Stage 2 proposal bundle without concept "
                    "inventory outer_fold=%s scope=%s inner_fold=%s path=%s",
                    outer_fold,
                    scope,
                    inner_fold,
                    path,
                )
                return None
            if self._concept_inventory_enabled() and not _concept_inventory_is_current(
                payload.get("concept_inventory"),
            ):
                logger.info(
                    "Ignoring cached Stage 2 proposal bundle with legacy concept "
                    "inventory outer_fold=%s scope=%s inner_fold=%s path=%s",
                    outer_fold,
                    scope,
                    inner_fold,
                    path,
                )
                return None
            payload["valid_proposals"] = [
                _proposal_from_dict(item)
                for item in payload.get("valid_proposals", [])
                if isinstance(item, dict)
            ]
            payload["resumed_from_cache"] = str(path)
            logger.info(
                "Reusing cached Stage 2 proposal bundle outer_fold=%s scope=%s "
                "inner_fold=%s path=%s",
                outer_fold,
                scope,
                inner_fold,
                path,
            )
            return payload
        except Exception as exc:
            logger.warning("Ignoring unreadable proposal bundle cache %s: %s", path, exc)
            return None

    def _write_proposal_bundle_cache(self, bundle: Dict[str, Any]) -> None:
        path = self._proposal_bundle_cache_path(
            outer_fold=int(bundle["outer_fold"]),
            scope=str(bundle["scope"]),
            inner_fold=bundle.get("inner_fold"),
        )
        _write_json(path, _proposal_bundle_artifact(bundle))

    def _inner_consistency_candidate_bundles(
        self,
        *,
        outer_fold: int,
        discovery_df: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        try:
            fold_count = _bounded_fold_count(
                int(self.nn_config.candidate_consistency_inner_folds),
                len(discovery_df),
            )
        except ValueError:
            return []

        splitter = KFold(
            n_splits=fold_count,
            shuffle=True,
            random_state=51_000 + int(outer_fold),
        )
        split_items = [
            (inner_fold, np.asarray(fit_pos), np.asarray(heldout_pos))
            for inner_fold, (fit_pos, heldout_pos) in enumerate(
                splitter.split(discovery_df),
                start=1,
            )
        ]
        n_jobs = self._candidate_consistency_n_jobs(len(split_items))
        if n_jobs > 1 and self._has_external_components:
            logger.warning(
                "Candidate consistency parallelism disabled because custom "
                "agent/extractor/evaluator objects were supplied and may not be "
                "thread-safe."
            )
            n_jobs = 1

        if n_jobs <= 1:
            return [
                self._build_inner_consistency_candidate_bundle(
                    outer_fold=outer_fold,
                    discovery_df=discovery_df,
                    inner_fold=int(inner_fold),
                    fit_pos=fit_pos,
                    heldout_pos=heldout_pos,
                    total_inner_folds=fold_count,
                )
                for inner_fold, fit_pos, heldout_pos in split_items
            ]

        logger.info(
            "Multi-model candidate consistency parallelism: outer_fold=%s "
            "inner_folds=%s n_jobs=%s setting=%s backend=%s joblib_backend=%s",
            outer_fold,
            len(split_items),
            n_jobs,
            self.nn_config.candidate_consistency_parallelism,
            self.nn_config.bow_parallel_backend,
            self._parallel_backend_name(),
        )
        return Parallel(
            n_jobs=n_jobs,
            backend=self._parallel_backend_name(),
            batch_size=1,
            pre_dispatch="all",
        )(
            delayed(_build_multi_model_inner_candidate_bundle_worker)(
                self.dataset,
                self.config,
                self.artifact_dir,
                int(outer_fold),
                discovery_df,
                int(inner_fold),
                fit_pos,
                heldout_pos,
                int(fold_count),
                self._inner_workers_for_nested_job(n_jobs),
            )
            for inner_fold, fit_pos, heldout_pos in split_items
        )

    def _build_inner_consistency_candidate_bundle_isolated(
        self,
        outer_fold: int,
        discovery_df: pd.DataFrame,
        inner_fold: int,
        fit_pos: np.ndarray,
        heldout_pos: np.ndarray,
        total_inner_folds: int,
        candidate_n_jobs: int,
    ) -> Dict[str, Any]:
        worker = MultiModelAgenticForestRunner(
            dataset=self.dataset,
            config=self.config,
            output_path=(
                self.artifact_dir
                / f"outer_{int(outer_fold):03d}_candidate_inner_{int(inner_fold):03d}"
                / "predictions.parquet"
            ),
            num_workers=self._inner_workers_for_nested_job(candidate_n_jobs),
        )
        return worker._build_inner_consistency_candidate_bundle(
            outer_fold=outer_fold,
            discovery_df=discovery_df,
            inner_fold=inner_fold,
            fit_pos=fit_pos,
            heldout_pos=heldout_pos,
            total_inner_folds=total_inner_folds,
        )

    def _build_inner_consistency_candidate_bundle(
        self,
        *,
        outer_fold: int,
        discovery_df: pd.DataFrame,
        inner_fold: int,
        fit_pos: np.ndarray,
        heldout_pos: np.ndarray,
        total_inner_folds: int,
    ) -> Dict[str, Any]:
        inner_df = discovery_df.iloc[np.asarray(fit_pos)].reset_index(drop=True)
        try:
            bow_result = self._fit_bow_discovery(
                inner_df,
                outer_fold=1000 * int(outer_fold) + int(inner_fold),
            )
            context = {
                **bow_result["context"],
                "outer_fold": int(outer_fold),
                "inner_fold": int(inner_fold),
                "consistency_scope": "inner_train",
                "inner_train_rows": int(len(fit_pos)),
                "inner_heldout_rows": int(len(heldout_pos)),
            }
            return self._propose_candidate_bundle(
                outer_fold=outer_fold,
                scope="inner_train",
                inner_fold=int(inner_fold),
                bow_context=context,
                n_rows=len(fit_pos),
                heldout_rows=len(heldout_pos),
            )
        except Exception as exc:
            logger.warning(
                "Skipping multi-model candidate consistency inner fold %s/%s "
                "for outer fold %s: %s",
                inner_fold,
                total_inner_folds,
                outer_fold,
                exc,
                exc_info=True,
            )
            return {
                "outer_fold": int(outer_fold),
                "scope": "inner_train",
                "inner_fold": int(inner_fold),
                "n_rows": int(len(fit_pos)),
                "heldout_rows": int(len(heldout_pos)),
                "error": str(exc),
                "valid_proposals": [],
                "rejected_proposals": [],
            }

    def _build_consistency_candidate_summaries(
        self,
        *,
        bundles: Sequence[Dict[str, Any]],
        alias_map: Dict[str, str],
        canonical_proposals: Dict[str, AgenticFeatureProposal],
    ) -> Tuple[List[Dict[str, Any]], int, int]:
        inner_folds = sorted(
            {
                int(bundle["inner_fold"])
                for bundle in bundles
                if bundle.get("scope") == "inner_train"
                and bundle.get("inner_fold") is not None
                and not bundle.get("error")
            }
        )
        inner_fold_count = len(inner_folds)
        threshold = _candidate_consistency_threshold(
            inner_fold_count,
            min_folds=int(self.nn_config.candidate_consistency_min_folds),
            min_fold_fraction=float(self.nn_config.candidate_consistency_min_fold_fraction),
        )

        summary_by_name: Dict[str, Dict[str, Any]] = {}
        for bundle in bundles:
            scope = str(bundle.get("scope") or "")
            inner_fold = bundle.get("inner_fold")
            for proposal in bundle.get("valid_proposals", []):
                if proposal.action != "add":
                    continue
                name = _resolve_alias_name(proposal.name, alias_map)
                canonical = canonical_proposals.get(name, proposal)
                summary = summary_by_name.setdefault(
                    name,
                    {
                        "name": name,
                        "type": canonical.type,
                        "categories": canonical.categories,
                        "roles": canonical.roles,
                        "description": canonical.description,
                        "expected_signal": canonical.expected_signal,
                        "inner_folds": [],
                        "proposed_on_full_outer_train": False,
                        "rationales": [],
                        "expected_signals": [],
                    },
                )
                summary["roles"] = _merge_ordered_values(
                    summary.get("roles"),
                    proposal.roles,
                )
                summary["categories"] = (
                    _merge_ordered_values(summary.get("categories"), proposal.categories) or None
                )
                summary["description"] = _merge_text_values(
                    summary.get("description"),
                    proposal.description,
                )
                summary["expected_signal"] = _merge_text_values(
                    summary.get("expected_signal"),
                    proposal.expected_signal,
                )
                if proposal.rationale:
                    summary["rationales"].append(
                        {
                            "scope": scope,
                            "inner_fold": inner_fold,
                            "text": proposal.rationale,
                        }
                    )
                if proposal.expected_signal:
                    summary["expected_signals"].append(str(proposal.expected_signal))
                if scope == "inner_train" and inner_fold is not None:
                    if int(inner_fold) not in summary["inner_folds"]:
                        summary["inner_folds"].append(int(inner_fold))
                elif scope == "full_outer_train":
                    summary["proposed_on_full_outer_train"] = True

        summaries = []
        for name in sorted(summary_by_name):
            summary = summary_by_name[name]
            support_count = len(summary["inner_folds"])
            support_fraction = (
                float(support_count / inner_fold_count) if inner_fold_count > 0 else None
            )
            summary["inner_folds"] = sorted(summary["inner_folds"])
            summary["inner_support_count"] = int(support_count)
            summary["inner_support_fraction"] = support_fraction
            summary["passes_consistency_gate"] = bool(
                support_count >= threshold
                or (inner_fold_count == 0 and summary["proposed_on_full_outer_train"])
            )
            summary["expected_signals"] = list(dict.fromkeys(summary["expected_signals"]))
            summaries.append(summary)
        return summaries, threshold, inner_fold_count

    def _build_consistency_context(
        self,
        *,
        outer_fold: int,
        candidate_summaries: List[Dict[str, Any]],
        threshold: int,
        inner_fold_count: int,
    ) -> Dict[str, Any]:
        recovery_limit = int(self.nn_config.candidate_consistency_recovery_max_candidates)
        below_threshold = [
            item
            for item in _rank_consistency_summaries(candidate_summaries)
            if not item.get("passes_consistency_gate")
        ]
        if len(below_threshold) > recovery_limit:
            raise ValueError(
                f"consistency review has {len(below_threshold)} below-threshold "
                f"candidates, exceeding configured "
                f"candidate_consistency_recovery_max_candidates={recovery_limit}; "
                "refusing silent candidate-summary omission"
            )
        passed = [
            item
            for item in _rank_consistency_summaries(candidate_summaries)
            if item.get("passes_consistency_gate")
        ]
        selection_limit = int(self.nn_config.candidate_proposals_per_fold)
        if len(passed) > selection_limit:
            raise ValueError(
                f"consistency review has {len(passed)} gate-passing candidates, "
                f"exceeding configured candidate_proposals_per_fold="
                f"{selection_limit}; refusing silent gate-passing candidate omission"
            )
        return {
            "prompt_version": "multi_model_agentic_consistency_v1",
            "outer_fold": int(outer_fold),
            "max_selected_candidates": int(self.nn_config.candidate_proposals_per_fold),
            "inner_fold_count": int(inner_fold_count),
            "min_support_folds": int(threshold),
            "min_support_fraction": float(self.nn_config.candidate_consistency_min_fold_fraction),
            "selection_policy": [
                "Return an exhaustive keep-list because omitted candidates are discarded downstream.",
                "Keep every candidate that passes the inner-fold support gate before considering any below-threshold recovery candidate.",
                "Recover below-threshold candidates only after all gate-passing candidates are kept and only when full outer-train evidence is strong or fold absence appears unstable rather than absent.",
                "Temporal eligibility is enforced upstream; do not reject supplied candidates based on treatment, response, outcome, survival, or toxicity semantics.",
                "Do not invent variables outside candidate_summaries.",
            ],
            "candidate_summaries": passed + below_threshold,
        }

    def _select_consistent_proposals(
        self,
        *,
        context: Dict[str, Any],
        candidate_summaries: Sequence[Dict[str, Any]],
        canonical_proposals: Dict[str, AgenticFeatureProposal],
    ) -> Tuple[List[AgenticFeatureProposal], Dict[str, Any]]:
        fallback_selected = _fallback_consistency_proposals(
            candidate_summaries,
            canonical_proposals,
        )
        max_selected = int(self.nn_config.candidate_proposals_per_fold)
        fallback_method = (
            "deterministic_consistency_gate"
            if any(
                item.get("passes_consistency_gate")
                for item in candidate_summaries
                if item.get("name") in {proposal.name for proposal in fallback_selected}
            )
            else "deterministic_full_outer_train_fallback"
        )
        try:
            raw_selection = self.proposal_agent.propose(context)
            agent_trace = _get_agent_response_trace(self.proposal_agent)
            selected, rejected = _agentic_consistency_selected_proposals(
                raw_selection,
                candidate_summaries=candidate_summaries,
                canonical_proposals=canonical_proposals,
                max_selected=max_selected,
            )
        except Exception as exc:
            logger.warning(
                "Multi-model consistency selection agent failed; using deterministic fallback",
                exc_info=True,
            )
            fallback_complete = _require_complete_consistency_fallback(
                fallback_selected,
                max_selected=max_selected,
            )
            return fallback_complete, {
                "selection_method": f"{fallback_method}_after_agent_error",
                "agent_selection_attempted": True,
                "agent_selection_used": False,
                "agent_error": str(exc),
                "max_selected_candidates": max_selected,
                "valid_proposals": [_proposal_to_dict(p) for p in fallback_complete],
                "rejected_proposals": [],
                "used_fallback": True,
            }

        if selected:
            artifact: Dict[str, Any] = {
                "selection_method": "agentic_consistency_selection",
                "agent_selection_attempted": True,
                "agent_selection_used": True,
                "max_selected_candidates": max_selected,
                "raw_proposals": raw_selection,
                "valid_proposals": [_proposal_to_dict(p) for p in selected],
                "rejected_proposals": rejected,
                "used_fallback": False,
            }
            if self.search_config.save_agent_raw_output:
                artifact["agent_raw_output"] = agent_trace
            return selected, artifact

        fallback_complete = _require_complete_consistency_fallback(
            fallback_selected,
            max_selected=max_selected,
        )
        artifact = {
            "selection_method": f"{fallback_method}_after_empty_agent_selection",
            "agent_selection_attempted": True,
            "agent_selection_used": False,
            "max_selected_candidates": max_selected,
            "raw_proposals": raw_selection,
            "agent_valid_proposals": [],
            "valid_proposals": [_proposal_to_dict(p) for p in fallback_complete],
            "rejected_proposals": rejected,
            "used_fallback": True,
        }
        if self.search_config.save_agent_raw_output:
            artifact["agent_raw_output"] = agent_trace
        return fallback_complete, artifact

    def _selected_specs_from_proposals(
        self,
        proposals: Sequence[AgenticFeatureProposal],
    ) -> List[ExplicitFeatureSpec]:
        return _dedupe_specs(
            [
                *self._initial_specs(),
                *[
                    ExplicitFeatureSpec(
                        name=proposal.name,
                        type=proposal.type or "continuous",
                        categories=proposal.categories,
                        roles=proposal.roles,
                        description=proposal.description,
                    )
                    for proposal in proposals
                    if proposal.action == "add"
                ],
            ]
        )

    def _resolve_proposal_aliases(
        self,
        *,
        outer_fold: int,
        proposals: List[AgenticFeatureProposal],
    ) -> Tuple[List[AgenticFeatureProposal], Dict[str, Any]]:
        add_proposals = [proposal for proposal in proposals if proposal.action == "add"]
        known_specs = _dedupe_specs(self.alias_reference_specs)
        if not add_proposals:
            return proposals, {"skipped": "no_valid_additions"}
        if len(add_proposals) < 2 and not known_specs:
            return proposals, {"skipped": "fewer_than_two_additions_and_no_known_features"}

        context = {
            "prompt_version": "multi_model_agentic_alias_resolution_v1",
            "outer_fold": int(outer_fold),
            "known_canonical_features": [_spec_to_dict(spec) for spec in known_specs],
            "proposed_features": [
                {
                    "name": proposal.name,
                    "type": proposal.type,
                    "categories": proposal.categories,
                    "roles": proposal.roles,
                    "description": proposal.description,
                    "rationale": proposal.rationale,
                    "expected_signal": proposal.expected_signal,
                }
                for proposal in add_proposals
            ],
        }

        try:
            response = self.proposal_agent.propose(context)
            alias_trace = _get_agent_response_trace(self.proposal_agent)
        except Exception as exc:
            logger.warning(
                "Multi-model alias resolution failed; using unmerged proposal names",
                exc_info=True,
            )
            return proposals, {"error": str(exc), "applied_aliases": []}

        resolved, applied_aliases = apply_agentic_alias_resolution(
            proposals=proposals,
            known_specs=known_specs,
            response=response,
        )
        result: Dict[str, Any] = {
            "response": response,
            "applied_aliases": applied_aliases,
        }
        if self.search_config.save_agent_raw_output:
            result["agent_raw_output"] = alias_trace
        return resolved, result

    def _harmonize_value_contracts(
        self,
        *,
        outer_fold: int,
        selected_specs: List[ExplicitFeatureSpec],
    ) -> Tuple[List[ExplicitFeatureSpec], Dict[str, Any]]:
        if not selected_specs:
            return selected_specs, {"skipped": "no_selected_features"}

        context = {
            "prompt_version": "multi_model_agentic_value_harmonization_v1",
            "outer_fold": int(outer_fold),
            "selected_features": [_spec_to_dict(spec) for spec in selected_specs],
            "missing_value_policy": (
                "Use null for unknown, not reported, not assessed, not tested, "
                "unavailable, and qualitative-only values that are incompatible "
                "with a numeric extraction target."
            ),
        }
        try:
            response = self.proposal_agent.propose(context)
            harmonization_trace = _get_agent_response_trace(self.proposal_agent)
        except Exception as exc:
            logger.warning(
                "Multi-model value harmonization failed; using unharmonized specs",
                exc_info=True,
            )
            return selected_specs, {"error": str(exc), "applied": []}

        harmonized, applied = apply_agentic_value_harmonization(
            specs=selected_specs,
            response=response,
        )
        result: Dict[str, Any] = {
            "response": response,
            "applied": applied,
        }
        if self.search_config.save_agent_raw_output:
            result["agent_raw_output"] = harmonization_trace
        return harmonized, result

    def _remember_alias_reference_specs(
        self,
        selected_specs: Sequence[ExplicitFeatureSpec],
    ) -> None:
        initial_names = {initial.name for initial in self._initial_specs()}
        self.alias_reference_specs = _dedupe_specs(
            [
                *self.alias_reference_specs,
                *[spec for spec in selected_specs if spec.name not in initial_names],
            ]
        )

    def _filter_specs_by_extraction_coverage(
        self,
        *,
        train_df: pd.DataFrame,
        specs: List[ExplicitFeatureSpec],
        outer_fold: Optional[int],
    ) -> List[ExplicitFeatureSpec]:
        initial_names = {spec.name for spec in self._initial_specs()}
        kept: List[ExplicitFeatureSpec] = []
        dropped: List[Dict[str, Any]] = []
        min_coverage = float(getattr(self.search_config, "min_feature_coverage", 0.0))
        for spec in specs:
            value_col = f"explicit_feat_{spec.name}"
            missing_col = f"{value_col}_missing"
            if value_col not in train_df.columns:
                coverage = 0.0
            elif missing_col in train_df.columns:
                coverage = float(1.0 - train_df[missing_col].astype(bool).mean())
            else:
                coverage = float(train_df[value_col].notna().mean())
            if spec.name in initial_names or coverage >= min_coverage:
                kept.append(spec)
                if outer_fold is not None:
                    self._clear_low_coverage_review_candidate(outer_fold, spec.name)
            else:
                dropped_item = {
                    "name": spec.name,
                    "coverage": coverage,
                    "required_min_coverage": min_coverage,
                    "feature": _spec_to_dict(spec),
                    "review_recommendation": (
                        "If the evidence still supports this clinical target, propose a "
                        "broader or more directly documented extraction target rather than "
                        "repeating the same low-coverage specification unchanged."
                    ),
                }
                dropped.append(dropped_item)
                self._remember_low_coverage_review_candidate(
                    outer_fold=outer_fold,
                    spec=spec,
                    coverage=coverage,
                    min_coverage=min_coverage,
                )
        if dropped:
            logger.info("Dropped low-coverage multi-model agentic features: %s", dropped)
            self.agent_rows.append(
                {
                    "outer_fold": None if outer_fold is None else int(outer_fold),
                    "event": "coverage_filter",
                    "dropped": dropped,
                }
            )
        return kept

    def _remember_low_coverage_review_candidate(
        self,
        *,
        outer_fold: Optional[int],
        spec: ExplicitFeatureSpec,
        coverage: float,
        min_coverage: float,
    ) -> None:
        if outer_fold is None:
            return
        by_name = self.low_coverage_review_candidates_by_outer.setdefault(int(outer_fold), {})
        by_name[spec.name] = {
            "name": spec.name,
            "coverage": float(coverage),
            "required_min_coverage": float(min_coverage),
            "feature": _spec_to_dict(spec),
            "review_recommendation": (
                "If the evidence still supports this clinical target, propose a broader "
                "or more directly documented extraction target rather than repeating the "
                "same low-coverage specification unchanged."
            ),
        }

    def _clear_low_coverage_review_candidate(
        self,
        outer_fold: int,
        name: str,
    ) -> None:
        by_name = self.low_coverage_review_candidates_by_outer.get(int(outer_fold))
        if not by_name:
            return
        by_name.pop(str(name), None)
        if not by_name:
            self.low_coverage_review_candidates_by_outer.pop(int(outer_fold), None)

    def _low_coverage_review_candidates(self, outer_fold: int) -> List[Dict[str, Any]]:
        by_name = self.low_coverage_review_candidates_by_outer.get(int(outer_fold), {})
        return sorted(
            [copy.deepcopy(item) for item in by_name.values()],
            key=lambda item: (float(item.get("coverage", 0.0)), str(item.get("name", ""))),
        )

    def _review_extracted_features_if_needed(
        self,
        *,
        outer_fold: int,
        train_idx: np.ndarray,
        selected_specs: List[ExplicitFeatureSpec],
        bow_result: Dict[str, Any],
        embedding_evidence: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not bool(getattr(self.nn_config, "extracted_feature_review_enabled", True)):
            return {
                "selected_specs": selected_specs,
                "summary": {
                    "enabled": False,
                    "review_passed": None,
                    "review_rounds": 0,
                },
            }

        max_rounds = int(getattr(self.nn_config, "extracted_feature_review_max_rounds", 3))
        if max_rounds <= 0:
            return {
                "selected_specs": selected_specs,
                "summary": {
                    "enabled": True,
                    "review_passed": None,
                    "review_rounds": 0,
                    "skipped": "max_rounds_zero",
                },
            }

        required_names = {spec.name for spec in self._initial_specs()}
        current_specs = list(selected_specs)
        best_specs = list(current_specs)
        best_diagnostic: Optional[Dict[str, Any]] = None
        best_score: Optional[Tuple[int, float, float, float]] = None
        final_status = "max_rounds_reached"
        final_passed = False

        for round_index in range(max_rounds + 1):
            train_df = self.dataset.iloc[train_idx].copy()
            current_specs = self._filter_specs_by_extraction_coverage(
                train_df=train_df,
                specs=current_specs,
                outer_fold=outer_fold,
            )
            diagnostic = _evaluate_extracted_feature_set_diagnostic(
                train_df=train_df,
                specs=current_specs,
                config=self.config,
                nn_config=self.nn_config,
                bow_metrics=bow_result.get("metrics", {}),
                embedding_evidence=embedding_evidence,
                random_state=71_000 + 100 * int(outer_fold) + int(round_index),
            )
            benchmark = diagnostic.get("benchmark", {})
            gate = _extracted_feature_review_gate(
                diagnostic=diagnostic,
                nn_config=self.nn_config,
            )
            diagnostic["outer_fold"] = int(outer_fold)
            diagnostic["round"] = int(round_index)
            diagnostic["selected_features"] = [_spec_to_dict(spec) for spec in current_specs]
            diagnostic["gate"] = gate
            self.extracted_feature_diagnostic_rows.append(
                _redact_review_artifact(diagnostic, self.search_config)
            )

            score = _extracted_review_selection_score(diagnostic, gate)
            if best_score is None or score < best_score:
                best_score = score
                best_specs = list(current_specs)
                best_diagnostic = diagnostic

            if gate.get("passed"):
                final_status = "passed"
                final_passed = True
                best_specs = list(current_specs)
                best_diagnostic = diagnostic
                break

            if round_index >= max_rounds:
                break

            context = self._build_extracted_feature_review_context(
                outer_fold=outer_fold,
                round_index=round_index,
                current_specs=current_specs,
                diagnostic=diagnostic,
                gate=gate,
                benchmark=benchmark,
                bow_context=bow_result["context"],
                embedding_evidence=embedding_evidence,
                htr_evidence=bow_result.get("htr_evidence") or {},
                required_names=required_names,
            )
            try:
                raw_proposals = self.proposal_agent.propose(context)
                review_agent_trace = _get_agent_response_trace(self.proposal_agent)
                proposals, rejected = validate_agentic_proposals(
                    raw_proposals,
                    current_specs=current_specs,
                    search_config=self.search_config,
                    allow_removals=True,
                    max_additions=self.nn_config.candidate_proposals_per_fold,
                )
            except Exception as exc:
                logger.warning(
                    "Multi-model extracted-feature review agent failed; "
                    "using best available feature set",
                    exc_info=True,
                )
                self.agent_rows.append(
                    {
                        "outer_fold": int(outer_fold),
                        "event": "extracted_feature_review",
                        "round": int(round_index),
                        "error": str(exc),
                    }
                )
                final_status = "agent_error"
                break

            proposals, protected_rejections = _protect_required_feature_proposals(
                proposals,
                required_names,
            )
            rejected.extend(protected_rejections)
            proposals, alias_resolution = self._resolve_proposal_aliases(
                outer_fold=outer_fold,
                proposals=proposals,
            )
            revised_specs = _dedupe_specs(apply_proposals(current_specs, proposals))
            revised_specs, value_harmonization = self._harmonize_value_contracts(
                outer_fold=outer_fold,
                selected_specs=revised_specs,
            )
            self._remember_alias_reference_specs(revised_specs)

            review_row: Dict[str, Any] = {
                "outer_fold": int(outer_fold),
                "event": "extracted_feature_review",
                "round": int(round_index),
                "raw_proposals": raw_proposals,
                "valid_proposals": [_proposal_to_dict(proposal) for proposal in proposals],
                "rejected_proposals": rejected,
                "alias_resolution": alias_resolution,
                "value_harmonization": value_harmonization,
                "selected_features_before": [_spec_to_dict(spec) for spec in current_specs],
                "selected_features_after": [_spec_to_dict(spec) for spec in revised_specs],
                "gate": gate,
            }
            if self.search_config.save_agent_context:
                review_row["context"] = context
            if self.search_config.save_agent_raw_output:
                review_row["agent_raw_output"] = review_agent_trace
            self.agent_rows.append(review_row)

            if not _spec_sets_differ(current_specs, revised_specs):
                final_status = "no_review_changes"
                break

            current_specs = revised_specs
            self._validate_complete_document_extraction(current_specs)
            self.dataset = self.extraction_provider.ensure_features(
                self.dataset,
                current_specs,
            )

        selected = best_specs
        if best_diagnostic is not None and not final_passed:
            selected = best_specs
        summary = _extracted_review_summary(
            diagnostic=best_diagnostic,
            status=final_status,
            passed=final_passed,
            rounds=len(
                [
                    row
                    for row in self.extracted_feature_diagnostic_rows
                    if row.get("outer_fold") == int(outer_fold)
                ]
            ),
        )
        return {"selected_specs": selected, "summary": summary}

    def _run_mandatory_parsimony_review(
        self,
        *,
        outer_fold: int,
        train_idx: np.ndarray,
        selected_specs: List[ExplicitFeatureSpec],
        bow_result: Dict[str, Any],
        embedding_evidence: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run value-driven cluster-to-factor parsimony before final fitting."""
        before_specs = list(selected_specs)
        required_names = {spec.name for spec in self._initial_specs()}
        if not bool(getattr(self.nn_config, "parsimony_review_enabled", False)):
            stop_reason = "disabled_by_config"
            summary = {
                "schema_version": _PARSIMONY_SCHEMA_VERSION,
                "strategy": "value_driven_cluster_factor",
                "enabled": False,
                "mandatory": False,
                "decision": "skipped",
                "stop_reason": stop_reason,
                "n_features_before": int(len(before_specs)),
                "n_features_after": int(len(before_specs)),
                "n_removed": 0,
                "removed_features": [],
                "added_factors": [],
                "n_clusters": 0,
                "n_factor_proposals": 0,
                "n_accepted_replacements": 0,
                "n_single_feature_ablations": 0,
            }
            review_row = {
                "schema_version": _PARSIMONY_SCHEMA_VERSION,
                "outer_fold": int(outer_fold),
                "event": "mandatory_parsimony_review",
                "strategy": "value_driven_cluster_factor",
                "decision": "skipped",
                "stop_reason": stop_reason,
                "required_features": sorted(required_names),
                "selected_features_before": [_spec_to_dict(spec) for spec in before_specs],
                "selected_features_after": [_spec_to_dict(spec) for spec in before_specs],
                "base_metrics": None,
                "base_gate": None,
                "redundancy_review": [],
                "ablations": [],
                "clusters": [],
                "factor_proposals": [],
                "replacement_evaluations": [],
                "summary": summary,
            }
            self.parsimony_review_rows.append(review_row)
            self.agent_rows.append(
                {
                    "outer_fold": int(outer_fold),
                    "event": "mandatory_parsimony_review",
                    "decision": "skipped",
                    "stop_reason": stop_reason,
                    "n_features_before": int(len(before_specs)),
                    "n_features_after": int(len(before_specs)),
                    "removed_features": [],
                    "artifact": "parsimony_review_by_fold.jsonl",
                }
            )
            return {"selected_specs": before_specs, "summary": summary}

        legacy_overrides = {
            "parsimony_review_auc_tolerance": 0.01,
            "parsimony_review_loss_relative_tolerance": 0.03,
            "parsimony_review_corr_threshold": 0.75,
            "parsimony_review_max_single_feature_ablations": 30,
        }
        changed_legacy = [
            name
            for name, default in legacy_overrides.items()
            if getattr(self.nn_config, name, default) != default
        ]
        if changed_legacy:
            logger.warning(
                "Ignoring deprecated single-feature parsimony settings: %s",
                changed_legacy,
            )

        train_df = self.dataset.iloc[train_idx].copy()
        diagnostic_seed = 91_000 + 100 * int(outer_fold)
        base_diagnostic = _evaluate_extracted_feature_set_diagnostic(
            train_df=train_df,
            specs=before_specs,
            config=self.config,
            nn_config=self.nn_config,
            bow_metrics=bow_result.get("metrics", {}),
            embedding_evidence=embedding_evidence,
            random_state=diagnostic_seed,
        )
        base_gate = _extracted_feature_review_gate(
            diagnostic=base_diagnostic,
            nn_config=self.nn_config,
        )
        base_diagnostic["gate"] = base_gate

        semantic_vectors, semantic_info = self._parsimony_semantic_vectors(before_specs)
        cluster_result = _build_value_driven_feature_clusters(
            train_df=train_df,
            specs=before_specs,
            semantic_vectors=semantic_vectors,
            nn_config=self.nn_config,
            random_state=diagnostic_seed + 7,
        )
        clusters = list(cluster_result.get("clusters", []))
        for cluster in clusters:
            self.parsimony_cluster_rows.append(
                {
                    "schema_version": _PARSIMONY_SCHEMA_VERSION,
                    "outer_fold": int(outer_fold),
                    "event": "value_driven_feature_cluster",
                    **cluster,
                }
            )

        contexts = [
            self._build_parsimony_factor_context(
                outer_fold=outer_fold,
                cluster=cluster,
                specs=before_specs,
                train_df=train_df,
                required_names=required_names,
                bow_result=bow_result,
                embedding_evidence=embedding_evidence,
            )
            for cluster in clusters
            if len(
                [
                    name
                    for name in cluster.get("member_names", [])
                    if name not in required_names
                ]
            )
            >= 2
        ]
        factor_agent_results = self._request_parsimony_factor_responses(contexts)
        candidates: List[Dict[str, Any]] = []
        for context, agent_result in zip(contexts, factor_agent_results):
            cluster = next(
                item for item in clusters if item.get("cluster_id") == context.get("cluster_id")
            )
            candidate, validation = _validate_parsimony_factor_candidate(
                response=agent_result.get("response"),
                context=context,
                cluster=cluster,
                current_specs=before_specs,
                required_names=required_names,
            )
            factor_row: Dict[str, Any] = {
                "schema_version": _PARSIMONY_SCHEMA_VERSION,
                "outer_fold": int(outer_fold),
                "event": "parsimony_factor_proposal",
                "cluster_id": context.get("cluster_id"),
                "context_fingerprint": _parsimony_context_fingerprint(context),
                "response": agent_result.get("response"),
                "validation": validation,
            }
            if agent_result.get("resumed_from_cache"):
                factor_row["resumed_from_cache"] = agent_result["resumed_from_cache"]
            if agent_result.get("error"):
                factor_row["agent_error"] = agent_result["error"]
            if self.search_config.save_agent_context:
                factor_row["context"] = context
            if self.search_config.save_agent_raw_output and agent_result.get("agent_raw_output"):
                factor_row["agent_raw_output"] = agent_result["agent_raw_output"]
            self.parsimony_factor_rows.append(factor_row)
            if candidate is not None:
                candidate["factor_row"] = factor_row
                candidates.append(candidate)

        alias_resolution: Dict[str, Any] = {"skipped": "no_valid_factor_proposals"}
        value_harmonization: Dict[str, Any] = {"skipped": "no_valid_factor_proposals"}
        if candidates:
            unique_factor_specs = _dedupe_specs(
                [spec for candidate in candidates for spec in candidate["factor_specs"]]
            )
            factor_proposals = [
                AgenticFeatureProposal(
                    action="add",
                    name=spec.name,
                    type=spec.type,
                    categories=spec.categories,
                    roles=list(spec.roles),
                    description=spec.description,
                    rationale="Operational factor proposed for value-cluster replacement.",
                    expected_signal="treatment, outcome, or pseudo-target signal expected",
                )
                for spec in unique_factor_specs
            ]
            resolved_proposals, alias_resolution = self._resolve_proposal_aliases(
                outer_fold=outer_fold,
                proposals=factor_proposals,
            )
            alias_map = {
                str(item.get("from")): str(item.get("to"))
                for item in alias_resolution.get("applied_aliases", [])
                if isinstance(item, dict) and item.get("from") and item.get("to")
            }
            resolved_specs = _dedupe_specs(
                [
                    ExplicitFeatureSpec(
                        name=proposal.name,
                        type=proposal.type or "continuous",
                        categories=proposal.categories,
                        roles=list(proposal.roles),
                        description=proposal.description,
                    )
                    for proposal in resolved_proposals
                    if proposal.action == "add"
                ]
            )
            resolved_specs, value_harmonization = self._harmonize_value_contracts(
                outer_fold=outer_fold,
                selected_specs=resolved_specs,
            )
            resolved_by_name = {spec.name: spec for spec in resolved_specs}
            viable_candidates: List[Dict[str, Any]] = []
            for candidate in candidates:
                names = _dedupe_strings(
                    [alias_map.get(spec.name, spec.name) for spec in candidate["factor_specs"]]
                )
                mapped_specs = [resolved_by_name[name] for name in names if name in resolved_by_name]
                expected_roles = {
                    role
                    for spec in before_specs
                    if spec.name in set(candidate["replaces"])
                    for role in spec.roles
                }
                observed_roles = {role for spec in mapped_specs for role in spec.roles}
                if not mapped_specs:
                    candidate["factor_row"]["post_harmonization_rejection"] = (
                        "no_factor_specs_after_alias_resolution"
                    )
                    continue
                if len(mapped_specs) >= len(set(candidate["replaces"])):
                    candidate["factor_row"]["post_harmonization_rejection"] = (
                        "factor_count_no_longer_reduces_spec_count"
                    )
                    continue
                if observed_roles != expected_roles:
                    candidate["factor_row"]["post_harmonization_rejection"] = {
                        "reason": "factor_role_union_changed_after_harmonization",
                        "expected": sorted(expected_roles),
                        "observed": sorted(observed_roles),
                    }
                    continue
                candidate["factor_specs"] = mapped_specs
                candidate["factor_row"]["resolved_factor_names"] = names
                viable_candidates.append(candidate)
            candidates = viable_candidates

        all_factor_specs = _dedupe_specs(
            [spec for candidate in candidates for spec in candidate["factor_specs"]]
        )
        # Persist expensive agent decisions before potentially long whole-note
        # factor extraction so interrupted folds can reuse identical contexts.
        self._flush_parsimony_fold_artifacts(outer_fold)
        if all_factor_specs:
            self._validate_complete_document_extraction(all_factor_specs)
            self.dataset = self.extraction_provider.ensure_features(
                self.dataset,
                all_factor_specs,
            )
            train_df = self.dataset.iloc[train_idx].copy()

        min_factor_coverage = max(
            float(getattr(self.nn_config, "parsimony_factor_min_coverage", 0.10)),
            float(getattr(self.search_config, "min_feature_coverage", 0.0)),
        )
        extraction_viable: List[Dict[str, Any]] = []
        for candidate in candidates:
            quality = _parsimony_factor_extraction_quality(
                train_df=train_df,
                factor_specs=candidate["factor_specs"],
                min_coverage=min_factor_coverage,
            )
            candidate["factor_row"]["extraction_quality"] = quality
            if quality.get("passed"):
                extraction_viable.append(candidate)
            else:
                candidate["factor_row"]["post_extraction_rejection"] = quality.get("reasons")
        candidates = extraction_viable

        self._flush_parsimony_fold_artifacts(outer_fold)

        def evaluate_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
            trial_specs = _apply_parsimony_factor_replacements(before_specs, [candidate])
            trial_diagnostic = _evaluate_extracted_feature_set_diagnostic(
                train_df=train_df,
                specs=trial_specs,
                config=self.config,
                nn_config=self.nn_config,
                bow_metrics=bow_result.get("metrics", {}),
                embedding_evidence=embedding_evidence,
                random_state=diagnostic_seed,
            )
            trial_gate = _extracted_feature_review_gate(
                diagnostic=trial_diagnostic,
                nn_config=self.nn_config,
            )
            trial_diagnostic["gate"] = trial_gate
            allowed, reasons, deltas = _strict_parsimony_replacement_decision(
                base_diagnostic=base_diagnostic,
                trial_diagnostic=trial_diagnostic,
                base_gate=base_gate,
                trial_gate=trial_gate,
                epsilon=float(getattr(self.nn_config, "parsimony_metric_epsilon", 1e-6)),
            )
            base_dim = _parsimony_model_dimension(base_diagnostic)
            trial_dim = _parsimony_model_dimension(trial_diagnostic)
            if trial_dim >= base_dim:
                allowed = False
                reasons = [*reasons, "expanded_model_dimension_not_reduced"]
            return {
                **candidate,
                "trial_specs": trial_specs,
                "diagnostic": trial_diagnostic,
                "gate": trial_gate,
                "allowed": bool(allowed),
                "reasons": _dedupe_strings(reasons),
                "metric_deltas": deltas,
                "base_dimension": int(base_dim),
                "trial_dimension": int(trial_dim),
                "dimension_reduction": int(base_dim - trial_dim),
            }

        n_jobs = self._parsimony_n_jobs(len(candidates))
        if n_jobs > 1:
            evaluated = Parallel(n_jobs=n_jobs, backend="threading", batch_size=1)(
                delayed(evaluate_candidate)(candidate) for candidate in candidates
            )
        else:
            evaluated = [evaluate_candidate(candidate) for candidate in candidates]

        passing: List[Dict[str, Any]] = []
        for result in evaluated:
            row = _parsimony_replacement_evaluation_row(
                outer_fold=outer_fold,
                phase="independent",
                result=result,
            )
            self.parsimony_evaluation_rows.append(row)
            if result.get("allowed"):
                passing.append(result)

        accepted: List[Dict[str, Any]] = []
        final_specs = list(before_specs)
        final_diagnostic = base_diagnostic
        final_gate = base_gate
        joint_result: Optional[Dict[str, Any]] = None
        if passing:
            joint_specs = _apply_parsimony_factor_replacements(before_specs, passing)
            if len(passing) == 1:
                joint_diagnostic = passing[0]["diagnostic"]
                joint_gate = passing[0]["gate"]
            else:
                joint_diagnostic = _evaluate_extracted_feature_set_diagnostic(
                    train_df=train_df,
                    specs=joint_specs,
                    config=self.config,
                    nn_config=self.nn_config,
                    bow_metrics=bow_result.get("metrics", {}),
                    embedding_evidence=embedding_evidence,
                    random_state=diagnostic_seed,
                )
                joint_gate = _extracted_feature_review_gate(
                    diagnostic=joint_diagnostic,
                    nn_config=self.nn_config,
                )
                joint_diagnostic["gate"] = joint_gate
            joint_allowed, joint_reasons, joint_deltas = (
                _strict_parsimony_replacement_decision(
                    base_diagnostic=base_diagnostic,
                    trial_diagnostic=joint_diagnostic,
                    base_gate=base_gate,
                    trial_gate=joint_gate,
                    epsilon=float(
                        getattr(self.nn_config, "parsimony_metric_epsilon", 1e-6)
                    ),
                )
            )
            if _parsimony_model_dimension(joint_diagnostic) >= _parsimony_model_dimension(
                base_diagnostic
            ):
                joint_allowed = False
                joint_reasons = [
                    *joint_reasons,
                    "expanded_model_dimension_not_reduced",
                ]
            joint_result = {
                "cluster_ids": [item["cluster_id"] for item in passing],
                "allowed": bool(joint_allowed),
                "reasons": _dedupe_strings(joint_reasons),
                "metric_deltas": joint_deltas,
                "base_dimension": _parsimony_model_dimension(base_diagnostic),
                "trial_dimension": _parsimony_model_dimension(joint_diagnostic),
                "dimension_reduction": (
                    _parsimony_model_dimension(base_diagnostic)
                    - _parsimony_model_dimension(joint_diagnostic)
                ),
                "diagnostic": joint_diagnostic,
                "gate": joint_gate,
                "trial_specs": joint_specs,
            }
            self.parsimony_evaluation_rows.append(
                _parsimony_replacement_evaluation_row(
                    outer_fold=outer_fold,
                    phase="joint",
                    result=joint_result,
                )
            )
            if joint_allowed:
                accepted = list(passing)
                final_specs = joint_specs
                final_diagnostic = joint_diagnostic
                final_gate = joint_gate
            else:
                ordered = sorted(
                    passing,
                    key=lambda item: (
                        -int(item.get("dimension_reduction", 0)),
                        -float(item.get("cluster", {}).get("empirical_cohesion", 0.0)),
                        str(item.get("cluster_id", "")),
                    ),
                )
                for candidate in ordered:
                    trial_candidates = [*accepted, candidate]
                    greedy_specs = _apply_parsimony_factor_replacements(
                        before_specs,
                        trial_candidates,
                    )
                    greedy_diagnostic = _evaluate_extracted_feature_set_diagnostic(
                        train_df=train_df,
                        specs=greedy_specs,
                        config=self.config,
                        nn_config=self.nn_config,
                        bow_metrics=bow_result.get("metrics", {}),
                        embedding_evidence=embedding_evidence,
                        random_state=diagnostic_seed,
                    )
                    greedy_gate = _extracted_feature_review_gate(
                        diagnostic=greedy_diagnostic,
                        nn_config=self.nn_config,
                    )
                    greedy_diagnostic["gate"] = greedy_gate
                    greedy_allowed, greedy_reasons, greedy_deltas = (
                        _strict_parsimony_replacement_decision(
                            base_diagnostic=base_diagnostic,
                            trial_diagnostic=greedy_diagnostic,
                            base_gate=base_gate,
                            trial_gate=greedy_gate,
                            epsilon=float(
                                getattr(
                                    self.nn_config,
                                    "parsimony_metric_epsilon",
                                    1e-6,
                                )
                            ),
                        )
                    )
                    if _parsimony_model_dimension(greedy_diagnostic) >= (
                        _parsimony_model_dimension(base_diagnostic)
                    ):
                        greedy_allowed = False
                        greedy_reasons = [
                            *greedy_reasons,
                            "expanded_model_dimension_not_reduced",
                        ]
                    greedy_result = {
                        "cluster_id": candidate["cluster_id"],
                        "cluster_ids": [item["cluster_id"] for item in trial_candidates],
                        "allowed": bool(greedy_allowed),
                        "reasons": _dedupe_strings(greedy_reasons),
                        "metric_deltas": greedy_deltas,
                        "base_dimension": _parsimony_model_dimension(base_diagnostic),
                        "trial_dimension": _parsimony_model_dimension(greedy_diagnostic),
                        "dimension_reduction": (
                            _parsimony_model_dimension(base_diagnostic)
                            - _parsimony_model_dimension(greedy_diagnostic)
                        ),
                        "diagnostic": greedy_diagnostic,
                        "gate": greedy_gate,
                        "trial_specs": greedy_specs,
                    }
                    self.parsimony_evaluation_rows.append(
                        _parsimony_replacement_evaluation_row(
                            outer_fold=outer_fold,
                            phase="greedy_backoff",
                            result=greedy_result,
                        )
                    )
                    if greedy_allowed:
                        accepted.append(candidate)
                        final_specs = greedy_specs
                        final_diagnostic = greedy_diagnostic
                        final_gate = greedy_gate

        removed = _dedupe_strings(
            [name for candidate in accepted for name in candidate.get("replaces", [])]
        )
        added_factors = _dedupe_strings(
            [spec.name for candidate in accepted for spec in candidate.get("factor_specs", [])]
        )
        decision = "replace_clusters" if accepted else "retain_all"
        if not before_specs:
            stop_reason = "no_selected_features"
        elif not clusters:
            stop_reason = "no_coherent_value_clusters"
        elif not candidates:
            stop_reason = "no_valid_extractable_factor_proposals"
        elif not passing:
            stop_reason = "no_replacement_preserved_all_metrics"
        elif not accepted:
            stop_reason = "joint_and_greedy_replacements_failed"
        else:
            stop_reason = "strict_cluster_replacements_accepted"
        summary = {
            "schema_version": _PARSIMONY_SCHEMA_VERSION,
            "strategy": "value_driven_cluster_factor",
            "enabled": True,
            "mandatory": True,
            "decision": decision,
            "stop_reason": stop_reason,
            "n_features_before": int(len(before_specs)),
            "n_features_after": int(len(final_specs)),
            "n_removed": int(len(removed)),
            "removed_features": removed,
            "added_factors": added_factors,
            "n_clusters": int(len(clusters)),
            "n_factor_proposals": int(len(factor_agent_results)),
            "n_valid_factor_proposals": int(len(candidates)),
            "n_independent_replacements_passing": int(len(passing)),
            "n_accepted_replacements": int(len(accepted)),
            "accepted_cluster_ids": [item["cluster_id"] for item in accepted],
            "n_single_feature_ablations": 0,
            **_prefix_metrics("final_", _parsimony_metric_snapshot(final_diagnostic)),
        }
        review_row = {
            "schema_version": _PARSIMONY_SCHEMA_VERSION,
            "outer_fold": int(outer_fold),
            "event": "mandatory_parsimony_review",
            "strategy": "value_driven_cluster_factor",
            "decision": decision,
            "stop_reason": stop_reason,
            "required_features": sorted(required_names),
            "selected_features_before": [_spec_to_dict(spec) for spec in before_specs],
            "selected_features_after": [_spec_to_dict(spec) for spec in final_specs],
            "base_metrics": _parsimony_metric_snapshot(base_diagnostic),
            "base_gate": base_gate,
            "cluster_generation": {
                **cluster_result.get("generation", {}),
                "semantic_encoding": semantic_info,
            },
            "clusters": [cluster.get("cluster_id") for cluster in clusters],
            "factor_proposals": [
                row.get("cluster_id")
                for row in self._outer_fold_rows(self.parsimony_factor_rows, outer_fold)
            ],
            "replacement_evaluations": int(
                len(self._outer_fold_rows(self.parsimony_evaluation_rows, outer_fold))
            ),
            "accepted_replacements": [
                {
                    "cluster_id": item["cluster_id"],
                    "replaces": list(item["replaces"]),
                    "factors": [_spec_to_dict(spec) for spec in item["factor_specs"]],
                }
                for item in accepted
            ],
            "alias_resolution": alias_resolution,
            "value_harmonization": value_harmonization,
            # Retain compatibility keys while making clear that no legacy
            # pairwise redundancy/ablation pass ran.
            "redundancy_review": {
                "strategy": "value_driven_sparse_neighbor_graph",
                "cluster_count": int(len(clusters)),
            },
            "ablations": [],
            "final_gate": final_gate,
            "summary": summary,
        }
        self.parsimony_review_rows.append(review_row)
        self.agent_rows.append(
            {
                "outer_fold": int(outer_fold),
                "event": "mandatory_parsimony_review",
                "decision": decision,
                "strategy": "value_driven_cluster_factor",
                "stop_reason": stop_reason,
                "n_features_before": int(len(before_specs)),
                "n_features_after": int(len(final_specs)),
                "removed_features": removed,
                "added_factors": added_factors,
                "artifact": "parsimony_review_by_fold.jsonl",
            }
        )
        self._flush_parsimony_fold_artifacts(outer_fold)
        return {"selected_specs": final_specs, "summary": summary}

    def _parsimony_semantic_vectors(
        self,
        specs: Sequence[ExplicitFeatureSpec],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        documents = [_parsimony_feature_contract_document(spec) for spec in specs]
        matrix, error = self._encode_concept_cluster_texts(documents)
        if matrix is not None:
            vectors = sklearn_normalize(np.asarray(matrix, dtype=float))
            return vectors, {"method": "embedding", "fallback_reason": None}
        vectors = _parsimony_tfidf_semantic_vectors(documents)
        return vectors, {"method": "tfidf_svd", "fallback_reason": error}

    def _build_parsimony_factor_context(
        self,
        *,
        outer_fold: int,
        cluster: Dict[str, Any],
        specs: Sequence[ExplicitFeatureSpec],
        train_df: pd.DataFrame,
        required_names: set,
        bow_result: Dict[str, Any],
        embedding_evidence: Dict[str, Any],
    ) -> Dict[str, Any]:
        member_names = list(cluster.get("member_names", []))
        member_set = set(member_names)
        member_specs = [spec for spec in specs if spec.name in member_set]
        replaceable = [spec.name for spec in member_specs if spec.name not in required_names]
        protected = [spec.name for spec in member_specs if spec.name in required_names]
        role_union = sorted({role for spec in member_specs if spec.name in replaceable for role in spec.roles})
        extraction_summary = {
            item["name"]: item
            for item in _summarize_multi_model_extractions(train_df, member_specs)
        }
        context: Dict[str, Any] = {
            "prompt_version": _PARSIMONY_FACTOR_PROMPT_VERSION,
            "schema_version": _PARSIMONY_SCHEMA_VERSION,
            "outer_fold": int(outer_fold),
            "cluster_id": cluster.get("cluster_id"),
            "clinical_question": self.config.clinical_question,
            "estimand": {
                "treatment_column": self.config.treatment_column,
                "outcome_column": self.config.outcome_column,
                "outcome_type": self.config.outcome_type,
            },
            "cluster": cluster,
            "cluster_members": [
                {
                    **_spec_to_dict(spec),
                    "protected": spec.name in required_names,
                    "extraction_summary": extraction_summary.get(spec.name, {}),
                }
                for spec in member_specs
            ],
            "replaceable_members": replaceable,
            "protected_members": protected,
            "required_role_union": role_union,
            "max_factors": int(
                getattr(self.nn_config, "parsimony_max_factors_per_cluster", 2)
            ),
            "temporal_policy": (
                "Use only evidence available before the treatment decision. The permitted "
                "source text is already temporally scoped; do not add an outcome-semantic ban."
            ),
            "response_contract": "cluster-to-factor operational rubric",
        }
        importance = (bow_result.get("context") or {}).get("feature_importance")
        if isinstance(importance, dict):
            context["bow_feature_evidence"] = _compact_multi_model_importance(importance)
        if embedding_evidence:
            context["embedding_contrast_evidence"] = _compact_embedding_contrast_evidence(
                embedding_evidence
            )
        htr_evidence = bow_result.get("htr_evidence") or {}
        if htr_evidence:
            context["htr_attention_evidence"] = _cluster_relevant_htr_evidence(
                htr_evidence,
                member_specs,
            )
        return _round_floats(context)

    def _request_parsimony_factor_responses(
        self,
        contexts: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not contexts:
            return []
        results: List[Optional[Dict[str, Any]]] = [None] * len(contexts)
        pending: List[Tuple[int, Dict[str, Any]]] = []
        cached_by_fingerprint: Dict[str, Dict[str, Any]] = {}
        outer_fold = int(contexts[0].get("outer_fold", 0) or 0)
        cache_path = (
            self.artifact_dir
            / f"outer_fold_{outer_fold:03d}"
            / "parsimony_factor_proposals_by_fold.jsonl"
        )
        if self.resume and cache_path.exists():
            for row in _read_jsonl(cache_path):
                fingerprint = str(row.get("context_fingerprint") or "")
                if fingerprint and row.get("response") is not None and not row.get("agent_error"):
                    cached_by_fingerprint[fingerprint] = row
        for index, context in enumerate(contexts):
            fingerprint = _parsimony_context_fingerprint(context)
            cached = cached_by_fingerprint.get(fingerprint)
            if cached is not None:
                results[index] = {
                    "response": cached.get("response"),
                    "agent_raw_output": cached.get("agent_raw_output"),
                    "resumed_from_cache": str(cache_path),
                }
            else:
                pending.append((index, context))
        if not pending:
            return [result or {"response": None} for result in results]

        n_jobs = self._parsimony_n_jobs(len(pending))
        if self._has_external_proposal_agent:
            n_jobs = 1
        if n_jobs <= 1:
            for index, context in pending:
                try:
                    response = self.proposal_agent.propose(context)
                    result: Dict[str, Any] = {"response": response}
                    trace = _get_agent_response_trace(self.proposal_agent)
                    if trace is not None:
                        result["agent_raw_output"] = trace
                except Exception as exc:
                    logger.warning(
                        "Parsimony factor agent failed for cluster %s; retaining cluster",
                        context.get("cluster_id"),
                        exc_info=True,
                    )
                    result = {
                        "response": None,
                        "error": f"{exc.__class__.__name__}: {exc}",
                    }
                results[index] = result
            return [result or {"response": None} for result in results]
        parallel_results = Parallel(n_jobs=n_jobs, backend="threading", batch_size=1)(
            delayed(_parsimony_factor_agent_worker)(self.search_config, context)
            for _, context in pending
        )
        for (index, _), result in zip(pending, parallel_results):
            results[index] = result
        return [result or {"response": None} for result in results]

    def _flush_parsimony_fold_artifacts(self, outer_fold: int) -> None:
        target_dir = self.artifact_dir / f"outer_fold_{int(outer_fold):03d}"
        target_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(
            target_dir / "parsimony_clusters_by_fold.jsonl",
            self._outer_fold_rows(self.parsimony_cluster_rows, outer_fold),
        )
        _write_jsonl(
            target_dir / "parsimony_factor_proposals_by_fold.jsonl",
            self._outer_fold_rows(self.parsimony_factor_rows, outer_fold),
        )
        _write_jsonl(
            target_dir / "parsimony_replacement_evaluations_by_fold.jsonl",
            self._outer_fold_rows(self.parsimony_evaluation_rows, outer_fold),
        )

    def _build_extracted_feature_review_context(
        self,
        *,
        outer_fold: int,
        round_index: int,
        current_specs: Sequence[ExplicitFeatureSpec],
        diagnostic: Dict[str, Any],
        gate: Dict[str, Any],
        benchmark: Dict[str, Any],
        bow_context: Dict[str, Any],
        embedding_evidence: Dict[str, Any],
        htr_evidence: Dict[str, Any],
        required_names: set,
    ) -> Dict[str, Any]:
        context = {
            "prompt_version": "multi_model_agentic_extracted_feature_review_v1",
            "outer_fold": int(outer_fold),
            "review_round": int(round_index),
            "max_proposals": int(self.nn_config.candidate_proposals_per_fold),
            "clinical_question": self.config.clinical_question,
            "estimand": {
                "treatment_column": self.config.treatment_column,
                "outcome_column": self.config.outcome_column,
                "outcome_type": self.config.outcome_type,
            },
            "required_features": [
                _spec_to_dict(spec) for spec in current_specs if spec.name in required_names
            ],
            "current_features": [_spec_to_dict(spec) for spec in current_specs],
            "extraction_summary": diagnostic.get("extraction_summary", []),
            "extracted_feature_diagnostics": _agent_visible_metrics(diagnostic.get("metrics", {})),
            "benchmarks": benchmark,
            "failed_criteria": gate.get("failed_criteria", []),
            "review_policy": {
                "auc_margin": float(
                    getattr(self.nn_config, "extracted_feature_review_auc_margin", 0.02)
                ),
                "loss_relative_margin": float(
                    getattr(
                        self.nn_config,
                        "extracted_feature_review_loss_relative_margin",
                        0.05,
                    )
                ),
                "min_benchmark_auc": float(
                    getattr(
                        self.nn_config,
                        "extracted_feature_review_min_benchmark_auc",
                        0.55,
                    )
                ),
                "low_coverage_feature_policy": (
                    "Features listed in low_coverage_features_needing_broader_targets "
                    "were proposed, extracted, and dropped for insufficient coverage. "
                    "Do not add the same target unchanged. If the original text evidence "
                    "still supports the concept, propose a broader or more directly "
                    "documented extraction target."
                ),
            },
            "original_bow_context": {
                "model_diagnostics": bow_context.get("model_diagnostics"),
                "feature_importance": bow_context.get("feature_importance"),
            },
            "response_contract": {
                "proposals": [
                    {
                        "action": "add|remove|update_role|none",
                        "name": "snake_case_variable_name",
                        "type": "categorical|continuous",
                        "categories": ["category_a", "category_b"],
                        "roles": ["confounder", "effect_modifier"],
                        "description": "exact extraction target represented in the supplied evidence",
                        "rationale": "why this change addresses the diagnostic failure",
                        "expected_signal": "treatment, outcome, or pseudo-target signal expected",
                    }
                ]
            },
        }
        low_coverage_candidates = self._low_coverage_review_candidates(outer_fold)
        if low_coverage_candidates:
            context["low_coverage_features_needing_broader_targets"] = low_coverage_candidates
        if embedding_evidence:
            context["embedding_contrast_evidence"] = (
                embedding_evidence
                if self.search_config.save_agent_context
                else redact_embedding_contrast_evidence(embedding_evidence)
            )
        if htr_evidence:
            context["htr_attention_evidence"] = htr_evidence
        return _compact_extracted_feature_review_context(context)

    def _build_candidate_signal_review_rows(
        self,
        *,
        outer_fold: int,
        train_df: pd.DataFrame,
        selected_specs: Sequence[ExplicitFeatureSpec],
        bow_result: Dict[str, Any],
        embedding_evidence: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for spec in selected_specs:
            diagnostic = _evaluate_extracted_feature_set_diagnostic(
                train_df=train_df,
                specs=[spec],
                config=self.config,
                nn_config=self.nn_config,
                bow_metrics=bow_result.get("metrics", {}),
                embedding_evidence=embedding_evidence,
                random_state=81_000 + 100 * int(outer_fold) + len(rows),
            )
            metrics = diagnostic.get("metrics", {})
            extraction_summary = (diagnostic.get("extraction_summary", [{}]) or [{}])[0]
            rows.append(
                {
                    "outer_fold": int(outer_fold),
                    "feature": spec.name,
                    "roles": list(spec.roles),
                    "type": spec.type,
                    "categories": spec.categories,
                    "description": spec.description,
                    "coverage": extraction_summary.get("coverage"),
                    "n_unique_observed": extraction_summary.get("n_unique_observed"),
                    "top_values": extraction_summary.get("top_values", {}),
                    "treatment_signal": {
                        "auroc": metrics.get("treatment_auroc"),
                        "brier": metrics.get("treatment_brier"),
                        "log_loss": metrics.get("treatment_log_loss"),
                    },
                    "outcome_signal": {
                        "auroc": metrics.get("outcome_auroc"),
                        "rmse": metrics.get("outcome_rmse"),
                        "brier": metrics.get("outcome_brier"),
                        "log_loss": metrics.get("outcome_log_loss"),
                    },
                    "r_signal": {
                        "r_loss_mean": metrics.get("r_loss_mean"),
                        "r_loss_relative_improvement": metrics.get("r_loss_relative_improvement"),
                        "tau_hat_pseudo_target_corr": metrics.get("tau_hat_pseudo_target_corr"),
                    },
                    "role_decision": _candidate_role_decision(spec),
                    "upstream_evidence": {
                        "bow_views": int(
                            len(
                                (bow_result.get("importance", {}) or {}).get(
                                    "views",
                                    [],
                                )
                            )
                        ),
                        "has_embedding_contrast": bool(embedding_evidence),
                        "has_htr_attention": bool(bow_result.get("htr_evidence")),
                    },
                }
            )
        return rows

    def _validate_complete_document_extraction(
        self,
        specs: Sequence[ExplicitFeatureSpec],
    ) -> None:
        if not specs or not bool(getattr(self.nn_config, "fail_on_extraction_truncation", True)):
            return
        if bool(getattr(self.extraction_provider, "reads_complete_documents", False)):
            return
        if not isinstance(self.extraction_provider, VLLMExplicitFeatureExtractionProvider):
            return
        max_chars = getattr(
            self.config.explicit_features,
            "extraction_max_text_length",
            None,
        )
        if max_chars is None:
            return
        max_chars = int(max_chars)
        if max_chars <= 0:
            return
        text_lengths = self.dataset[self.config.text_column].fillna("").astype(str).str.len()
        too_long = text_lengths > max_chars
        if not bool(too_long.any()):
            return
        raise ValueError(
            "multi_model_agentic_forest final feature extraction requires "
            "complete-document reading. The built-in VLLM extractor would "
            f"truncate {int(too_long.sum())} note(s) because "
            f"explicit_features.extraction_max_text_length={max_chars}. "
            "Increase extraction_max_text_length, set it to null, use a "
            "complete-document recursive extraction provider, or set "
            "multi_model_agentic_forest.fail_on_extraction_truncation=False "
            "only for non-skill-compatible debugging runs."
        )

    def _ensure_prespecified_features(self) -> None:
        specs = self._initial_specs()
        if not specs:
            return
        self._validate_complete_document_extraction(specs)
        self.dataset = self.extraction_provider.ensure_features(self.dataset, specs)

    def _initial_specs(self) -> List[ExplicitFeatureSpec]:
        specs: List[ExplicitFeatureSpec] = []
        if getattr(self.config.explicit_features, "features", None):
            specs.extend(list(self.config.explicit_features.features))
        specs.extend(list(getattr(self.nn_config, "prespecified_features", []) or []))
        specs.extend(list(getattr(self.nn_config, "prespecified_confounders", []) or []))
        specs.extend(list(getattr(self.nn_config, "prespecified_effect_modifiers", []) or []))
        json_path = getattr(self.nn_config, "prespecified_features_json", None)
        if json_path:
            specs.extend(load_explicit_feature_specs_json(str(json_path)))
        return _dedupe_specs(specs)

    def _vectorizer_params(self, view: BoWViewConfig) -> Dict[str, Any]:
        return _bow_vectorizer_params(view)

    def _model_params(self, view: BoWViewConfig) -> Dict[str, Any]:
        return _bow_model_params(view)

    def _make_vectorizer(self, view: BoWViewConfig) -> TfidfVectorizer:
        return _make_bow_vectorizer(self._vectorizer_params(view))

    def _make_classifier(self, view: BoWViewConfig, random_state: int = 17):
        return _make_bow_classifier(self._model_params(view), random_state=random_state)

    def _make_regressor(self, view: BoWViewConfig, random_state: int = 17):
        return _make_bow_regressor(self._model_params(view), random_state=random_state)

    def _make_logistic_regression(
        self,
        view: BoWViewConfig,
        random_state: int = 17,
    ) -> LogisticRegression:
        if view.bow_model != "linear":
            raise ValueError("logistic helper requires a linear BoW view")
        return _make_bow_classifier(
            _bow_model_params(view),
            random_state=random_state,
        )

    def _make_ridge(self, view: BoWViewConfig) -> Ridge:
        if view.bow_model != "linear":
            raise ValueError("ridge helper requires a linear BoW view")
        return _make_bow_regressor(
            _bow_model_params(view),
            random_state=17,
        )

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

    def _candidate_consistency_n_jobs(self, folds: int) -> int:
        return self._parallel_n_jobs(
            self.nn_config.candidate_consistency_parallelism,
            folds,
            auto_workers=self.num_workers,
        )

    def _parsimony_n_jobs(self, tasks: int) -> int:
        return self._parallel_n_jobs(
            self.nn_config.parsimony_parallelism,
            tasks,
            auto_workers=self.num_workers,
        )

    def _inner_workers_for_outer_job(self, outer_n_jobs: int) -> int:
        if str(self.nn_config.fold_parallelism).strip().lower() != "auto":
            return self.num_workers
        return max(1, int(self.num_workers) // max(1, int(outer_n_jobs)))

    def _inner_workers_for_nested_job(self, n_jobs: int) -> int:
        if str(self.nn_config.fold_parallelism).strip().lower() != "auto":
            return self.num_workers
        return max(1, int(self.num_workers) // max(1, int(n_jobs)))

    def _fold_n_jobs(self, folds: int) -> int:
        return self._parallel_n_jobs(
            self.nn_config.fold_parallelism,
            folds,
            auto_workers=self.num_workers,
        )

    def _feature_importance_n_jobs(self) -> int:
        return self._parallel_n_jobs(
            self.nn_config.fold_parallelism,
            3,
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
        backend = self._parallel_backend_name()
        logger.info(
            "Multi-model BoW cross-fit parallelism: folds=%s n_jobs=%s "
            "setting=%s backend=%s joblib_backend=%s",
            len(split_items),
            n_jobs,
            self.nn_config.fold_parallelism,
            self.nn_config.bow_parallel_backend,
            backend,
        )
        return Parallel(
            n_jobs=n_jobs,
            backend=backend,
            batch_size=1,
            pre_dispatch="all",
        )(
            delayed(run_fold)(int(fold), np.asarray(fit_pos), np.asarray(heldout_pos))
            for fold, (fit_pos, heldout_pos) in split_items
        )

    def _dataset_summary(self) -> Dict[str, Any]:
        df = self.dataset
        text_lengths = df[self.config.text_column].fillna("").astype(str).str.len()
        summary: Dict[str, Any] = {
            "n_rows": int(len(df)),
            "row_id_column": "_oci_row_id",
            "text_column": self.config.text_column,
            "treatment_column": self.config.treatment_column,
            "outcome_column": self.config.outcome_column,
            "outcome_type": self.config.outcome_type,
            "chronology_assumption": (
                "clinical_text is treated as baseline/pre-treatment/pre-outcome "
                "unless the user configuration states otherwise"
            ),
            "missingness": {
                column: float(df[column].isna().mean())
                for column in [
                    self.config.text_column,
                    self.config.treatment_column,
                    self.config.outcome_column,
                ]
                if column in df.columns
            },
            "note_length_chars": {
                "min": _finite_or_none(text_lengths.min()),
                "median": _finite_or_none(text_lengths.median()),
                "mean": _finite_or_none(text_lengths.mean()),
                "max": _finite_or_none(text_lengths.max()),
            },
        }
        if self.config.treatment_column in df.columns:
            t = pd.to_numeric(df[self.config.treatment_column], errors="coerce")
            summary["treatment_rate"] = _finite_or_none(t.mean())
        if self.config.outcome_column in df.columns:
            y = pd.to_numeric(df[self.config.outcome_column], errors="coerce")
            summary["outcome_rate_or_mean"] = _finite_or_none(y.mean())
        if {
            self.config.treatment_column,
            self.config.outcome_column,
        }.issubset(df.columns):
            table = pd.crosstab(
                df[self.config.treatment_column],
                df[self.config.outcome_column],
                dropna=False,
            )
            summary["treatment_outcome_table"] = {
                str(treatment): {str(outcome): int(count) for outcome, count in row.items()}
                for treatment, row in table.iterrows()
            }
        return summary

    def _report_text(self) -> str:
        summary = self._dataset_summary()
        honest_rows = [row for row in self.split_provenance_rows if row.get("honest_outer_holdout")]
        lines = [
            "Multi-Model Agentic Forest Skill-Aligned Report",
            "",
            "Dataset",
            f"- Rows: {summary.get('n_rows')}",
            f"- Text column: {summary.get('text_column')}",
            f"- Treatment column: {summary.get('treatment_column')}",
            f"- Outcome column: {summary.get('outcome_column')} ({summary.get('outcome_type')})",
            f"- Treatment rate: {summary.get('treatment_rate')}",
            f"- Outcome rate/mean: {summary.get('outcome_rate_or_mean')}",
            f"- Note length chars: {summary.get('note_length_chars')}",
            f"- Chronology: {summary.get('chronology_assumption')}",
            "",
            "Folds",
            f"- Outer split rows: {len(self.split_provenance_rows)}",
            f"- Honest outer holdout folds: {len(honest_rows)}",
            "",
            "Evidence And Review",
            f"- Feature discovery methods: {self._enabled_feature_discovery_methods()}",
            f"- BoW evidence records: {len(self.importance_rows)}",
            f"- Embedding evidence records: {len(self.embedding_evidence_rows)}",
            f"- HTR attention records: {len(self.htr_attention_rows)}",
            f"- Candidate proposal/review records: {len(self.agent_rows)}",
            f"- Extracted-feature diagnostic records: {len(self.extracted_feature_diagnostic_rows)}",
            f"- Candidate signal review records: {len(self.candidate_signal_review_rows)}",
            f"- Parsimony review records: {len(self.parsimony_review_rows)}",
            f"- Value-cluster records: {len(self.parsimony_cluster_rows)}",
            f"- Factor proposal records: {len(self.parsimony_factor_rows)}",
            f"- Replacement evaluation records: {len(self.parsimony_evaluation_rows)}",
            "",
            "Final Variables",
        ]
        for row in self.feature_set_rows:
            lines.append(
                "- outer_fold={fold}: features={features}; confounders={conf}; "
                "effect_modifiers={mods}".format(
                    fold=row.get("outer_fold"),
                    features=[
                        item.get("name")
                        for item in row.get("selected_features", [])
                        if isinstance(item, dict)
                    ],
                    conf=row.get("confounders", []),
                    mods=row.get("effect_modifiers", []),
                )
            )
        lines.extend(
            [
                "",
                "Artifacts",
                "- text_evidence.bow.jsonl, text_evidence.embedding.jsonl, text_evidence.htr.parquet",
                "- ensemble_nuisance_predictions.parquet",
                "- candidate_features.parquet and candidate_signal_review.jsonl",
                "- parsimony_review.by_fold.jsonl",
                "- parsimony_clusters_by_fold.jsonl",
                "- parsimony_factor_proposals_by_fold.jsonl",
                "- parsimony_replacement_evaluations_by_fold.jsonl",
                "- ite_estimates.parquet",
            ]
        )
        return "\n".join(lines) + "\n"

    def _save_predictions(self, results_df: pd.DataFrame) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.prediction_results = results_df.copy()
        results_df.to_parquet(self.output_path, index=False)
        logger.info("Multi-model agentic forest predictions saved to: %s", self.output_path)

    @staticmethod
    def _outer_fold_rows(
        rows: Sequence[Dict[str, Any]],
        outer_fold: int,
    ) -> List[Dict[str, Any]]:
        fold = int(outer_fold)
        return [row for row in rows if row.get("outer_fold") == fold]

    def _save_outer_fold_checkpoint(
        self,
        *,
        outer_fold: int,
        predictions: Optional[pd.DataFrame],
        target_dir: Path,
    ) -> None:
        """Persist fold-local trace artifacts as soon as a fold completes."""
        fold = int(outer_fold)
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        if predictions is not None:
            predictions.to_parquet(target_dir / "predictions.parquet", index=False)

        selected_rows = self._outer_fold_rows(self.feature_set_rows, fold)
        _write_json(target_dir / "selected_feature_sets.json", selected_rows)
        if selected_rows:
            _write_json(target_dir / "selected_feature_set.json", selected_rows[-1])

        agent_rows = self._outer_fold_rows(self.agent_rows, fold)
        diagnostics = self._outer_fold_rows(
            self.extracted_feature_diagnostic_rows,
            fold,
        )
        signal_rows = self._outer_fold_rows(self.candidate_signal_review_rows, fold)
        parsimony_rows = self._outer_fold_rows(self.parsimony_review_rows, fold)
        parsimony_cluster_rows = self._outer_fold_rows(self.parsimony_cluster_rows, fold)
        parsimony_factor_rows = self._outer_fold_rows(self.parsimony_factor_rows, fold)
        parsimony_evaluation_rows = self._outer_fold_rows(
            self.parsimony_evaluation_rows,
            fold,
        )
        metric_rows = self._outer_fold_rows(self.outer_metric_rows, fold)
        split_rows = self._outer_fold_rows(self.split_provenance_rows, fold)
        importance_rows = self._outer_fold_rows(self.importance_rows, fold)
        embedding_rows = self._outer_fold_rows(self.embedding_evidence_rows, fold)

        _write_jsonl(target_dir / "agent_candidate_proposals.jsonl", agent_rows)
        _write_jsonl(
            target_dir / "extracted_feature_diagnostics_by_fold.jsonl",
            diagnostics,
        )
        _write_jsonl(target_dir / "candidate_signal_review.jsonl", signal_rows)
        _write_jsonl(target_dir / "parsimony_review_by_fold.jsonl", parsimony_rows)
        _write_jsonl(target_dir / "parsimony_review.by_fold.jsonl", parsimony_rows)
        _write_jsonl(
            target_dir / "parsimony_clusters_by_fold.jsonl",
            parsimony_cluster_rows,
        )
        _write_jsonl(
            target_dir / "parsimony_factor_proposals_by_fold.jsonl",
            parsimony_factor_rows,
        )
        _write_jsonl(
            target_dir / "parsimony_replacement_evaluations_by_fold.jsonl",
            parsimony_evaluation_rows,
        )
        _write_jsonl(target_dir / "split_provenance.jsonl", split_rows)
        _write_jsonl(
            target_dir / "bow_view_feature_importance_by_fold.jsonl",
            importance_rows,
        )
        _write_jsonl(target_dir / "text_evidence.bow.jsonl", importance_rows)
        _write_jsonl(
            target_dir / "embedding_contrast_evidence_by_fold.jsonl",
            embedding_rows,
        )
        _write_jsonl(target_dir / "text_evidence.embedding.jsonl", embedding_rows)
        pd.DataFrame(metric_rows).to_csv(target_dir / "outer_cv_metrics.csv", index=False)

        _write_json(
            target_dir / "checkpoint_summary.json",
            {
                "outer_fold": fold,
                "parsimony_schema_version": _PARSIMONY_SCHEMA_VERSION,
                "n_predictions": 0 if predictions is None else int(len(predictions)),
                "n_selected_feature_rows": int(len(selected_rows)),
                "n_agent_rows": int(len(agent_rows)),
                "n_extracted_feature_diagnostic_rows": int(len(diagnostics)),
                "n_candidate_signal_review_rows": int(len(signal_rows)),
                "n_parsimony_review_rows": int(len(parsimony_rows)),
                "n_parsimony_cluster_rows": int(len(parsimony_cluster_rows)),
                "n_parsimony_factor_rows": int(len(parsimony_factor_rows)),
                "n_parsimony_evaluation_rows": int(len(parsimony_evaluation_rows)),
                "n_metric_rows": int(len(metric_rows)),
            },
        )
        logger.info(
            "Multi-model outer fold %s checkpoint artifacts saved to: %s",
            fold,
            target_dir,
        )

    def _save_artifacts(self) -> None:
        bow_predictions = (
            pd.concat(self.bow_prediction_frames, ignore_index=True)
            if self.bow_prediction_frames
            else pd.DataFrame()
        )
        if not bow_predictions.empty:
            bow_predictions.to_parquet(
                self.artifact_dir / "bow_view_oof_predictions.parquet",
                index=False,
            )
        text_prediction_frames = []
        if not bow_predictions.empty:
            text_prediction_frames.append(bow_predictions)
        if self.htr_nuisance_prediction_frames:
            htr_nuisance = pd.concat(
                self.htr_nuisance_prediction_frames,
                ignore_index=True,
            )
            htr_nuisance.to_parquet(
                self.artifact_dir / "htr_nuisance_oof_predictions.parquet",
                index=False,
            )
            text_prediction_frames.append(htr_nuisance)
        if self.htr_effect_prediction_frames:
            htr_effect = pd.concat(
                self.htr_effect_prediction_frames,
                ignore_index=True,
            )
            htr_effect.to_parquet(
                self.artifact_dir / "htr_effect_oof_predictions.parquet",
                index=False,
            )
            text_prediction_frames.append(htr_effect)
        if text_prediction_frames:
            pd.concat(text_prediction_frames, ignore_index=True).to_parquet(
                self.artifact_dir / "text_model_oof_predictions.parquet",
                index=False,
            )
        if self.htr_attention_rows:
            htr_attention = pd.DataFrame(self.htr_attention_rows)
            htr_attention.to_parquet(
                self.artifact_dir / "htr_attention_evidence.parquet",
                index=False,
            )
            htr_attention.to_parquet(
                self.artifact_dir / "text_evidence.htr.parquet",
                index=False,
            )
        ensemble_nuisance = _ensemble_nuisance_artifact_frame(
            source_frames=text_prediction_frames,
            ensemble_frames=self.ensemble_nuisance_prediction_frames,
        )
        if not ensemble_nuisance.empty:
            ensemble_nuisance.to_parquet(
                self.artifact_dir / "ensemble_nuisance_predictions.parquet",
                index=False,
            )
        pd.DataFrame(self.outer_metric_rows).to_csv(
            self.artifact_dir / "outer_cv_metrics.csv",
            index=False,
        )
        _write_jsonl(
            self.artifact_dir / "split_provenance.jsonl",
            self.split_provenance_rows,
        )
        with open(self.artifact_dir / "dataset_summary.json", "w") as f:
            json.dump(self._dataset_summary(), f, indent=2, default=_json_default)
        _write_jsonl(
            self.artifact_dir / "bow_view_feature_importance_by_fold.jsonl",
            self.importance_rows,
        )
        _write_jsonl(
            self.artifact_dir / "text_evidence.bow.jsonl",
            self.importance_rows,
        )
        if self.embedding_evidence_rows:
            _write_jsonl(
                self.artifact_dir / "embedding_contrast_evidence_by_fold.jsonl",
                self.embedding_evidence_rows,
            )
            _write_jsonl(
                self.artifact_dir / "text_evidence.embedding.jsonl",
                self.embedding_evidence_rows,
            )
        if self.extracted_feature_diagnostic_rows:
            _write_jsonl(
                self.artifact_dir / "extracted_feature_diagnostics_by_fold.jsonl",
                self.extracted_feature_diagnostic_rows,
            )
        if self.candidate_signal_review_rows:
            _write_jsonl(
                self.artifact_dir / "candidate_signal_review.jsonl",
                self.candidate_signal_review_rows,
            )
        if self.parsimony_review_rows:
            _write_jsonl(
                self.artifact_dir / "parsimony_review_by_fold.jsonl",
                self.parsimony_review_rows,
            )
            _write_jsonl(
                self.artifact_dir / "parsimony_review.by_fold.jsonl",
                self.parsimony_review_rows,
            )
        if self.parsimony_cluster_rows:
            _write_jsonl(
                self.artifact_dir / "parsimony_clusters_by_fold.jsonl",
                self.parsimony_cluster_rows,
            )
        if self.parsimony_factor_rows:
            _write_jsonl(
                self.artifact_dir / "parsimony_factor_proposals_by_fold.jsonl",
                self.parsimony_factor_rows,
            )
        if self.parsimony_evaluation_rows:
            _write_jsonl(
                self.artifact_dir / "parsimony_replacement_evaluations_by_fold.jsonl",
                self.parsimony_evaluation_rows,
            )
        _write_jsonl(self.artifact_dir / "agent_candidate_proposals.jsonl", self.agent_rows)
        with open(self.artifact_dir / "selected_feature_sets.json", "w") as f:
            json.dump(self.feature_set_rows, f, indent=2, default=_json_default)
        candidate_features = _candidate_features_frame(self.dataset)
        candidate_features.to_parquet(
            self.artifact_dir / "candidate_features.parquet",
            index=False,
        )
        with open(self.artifact_dir / "candidate_features.specs.json", "w") as f:
            json.dump(self.feature_set_rows, f, indent=2, default=_json_default)
        if self.prediction_results is not None:
            self.prediction_results.to_parquet(
                self.artifact_dir / "ite_estimates.parquet",
                index=False,
            )
        with open(self.artifact_dir / "report.txt", "w") as f:
            f.write(self._report_text())
        logger.info("Multi-model agentic forest artifacts saved to: %s", self.artifact_dir)


def _required_applied_seed(config: AppliedInferenceConfig) -> int:
    """Return the explicitly supplied scientific seed or fail closed."""

    seed = config.seed
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31:
        raise ValueError(
            "applied inference requires an explicit integer seed in [0, 2**31) "
            "for physical-fit identity"
        )
    return int(seed)


def _agentic_discovery_handoff_row(
    result: Dict[str, Any],
    *,
    fold_key: int,
    outer_fold: int,
    scope: str,
    n_rows: int,
    inner_fold: Optional[int] = None,
    heldout_rows: Optional[int] = None,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "schema_version": "multi_model_agentic_discovery_handoff_v1",
        "fold_key": int(fold_key),
        "outer_fold": int(outer_fold),
        "scope": str(scope),
        "n_rows": int(n_rows),
        "metrics": result.get("metrics") or {},
        "importance": result.get("importance") or {},
        "embedding_contrast_evidence": result.get("embedding_contrast_evidence") or {},
        "context": result.get("context") or {},
    }
    htr_evidence = result.get("htr_evidence") or {}
    if htr_evidence:
        row["htr_evidence"] = htr_evidence
    if inner_fold is not None:
        row["inner_fold"] = int(inner_fold)
    if heldout_rows is not None:
        row["heldout_rows"] = int(heldout_rows)
    return row


def _load_agentic_discovery_handoff(path: Path) -> Dict[int, Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Precomputed agentic discovery handoff not found: {path}")
    rows: Dict[int, Dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            fold_key = int(row.get("fold_key", row.get("outer_fold")))
            if fold_key in rows:
                raise ValueError(
                    f"Duplicate fold_key={fold_key} in agentic discovery handoff {path}"
                )
            rows[fold_key] = row
            if row.get("schema_version") != "multi_model_agentic_discovery_handoff_v1":
                raise ValueError(
                    f"Unsupported agentic discovery handoff schema on line {line_number}: "
                    f"{row.get('schema_version')!r}"
                )
    if not rows:
        raise ValueError(f"Agentic discovery handoff is empty: {path}")
    return rows


class PrecomputedDiscoveryMultiModelAgenticForestRunner(MultiModelAgenticForestRunner):
    """Agentic runner that consumes saved discovery evidence instead of refitting it."""

    def __init__(
        self,
        *,
        handoff_path: Path,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            embedding_provider=object(),
            htr_evidence_provider=object(),
            **kwargs,
        )
        self.handoff_path = Path(handoff_path)
        self._handoff_by_key = _load_agentic_discovery_handoff(self.handoff_path)

    def _fit_bow_discovery(
        self,
        discovery_df: pd.DataFrame,
        outer_fold: int,
    ) -> Dict[str, Any]:
        del discovery_df
        fold_key = int(outer_fold)
        row = self._handoff_by_key.get(fold_key)
        if row is None:
            raise RuntimeError(
                "Missing precomputed agentic discovery handoff for fold_key="
                f"{fold_key} in {self.handoff_path}. Rerun multi_model_forest "
                "Stage 1 to regenerate the handoff."
            )
        logger.info(
            "Using precomputed agentic discovery handoff fold_key=%s scope=%s path=%s",
            fold_key,
            row.get("scope"),
            self.handoff_path,
        )
        return {
            "predictions": pd.DataFrame(columns=["_oci_row_id", "outer_fold"]),
            "metrics": copy.deepcopy(row.get("metrics") or {}),
            "importance": copy.deepcopy(row.get("importance") or {}),
            "embedding_contrast_evidence": copy.deepcopy(
                row.get("embedding_contrast_evidence") or {}
            ),
            "htr_evidence": copy.deepcopy(row.get("htr_evidence") or {}),
            "context": copy.deepcopy(row.get("context") or {}),
        }

    def _candidate_consistency_n_jobs(self, folds: int) -> int:
        del folds
        return 1


def _run_multi_model_outer_fold_worker(
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    artifact_dir: Path,
    outer_fold: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    num_workers: int,
) -> Dict[str, Any]:
    logger.info(
        "Multi-model agentic isolated fold %s: train=%s test=%s workers=%s",
        outer_fold,
        len(train_idx),
        len(test_idx),
        num_workers,
    )
    fold_runner = MultiModelAgenticForestRunner(
        dataset=dataset,
        config=config,
        output_path=(
            Path(artifact_dir) / f"outer_fold_{int(outer_fold):03d}" / "predictions.parquet"
        ),
        num_workers=num_workers,
    )
    predictions = fold_runner._run_one_analysis_split(
        outer_fold=outer_fold,
        train_idx=train_idx,
        test_idx=test_idx,
    )
    fold_runner._save_outer_fold_checkpoint(
        outer_fold=int(outer_fold),
        predictions=predictions,
        target_dir=Path(artifact_dir) / f"outer_fold_{int(outer_fold):03d}",
    )
    return {
        "outer_fold": int(outer_fold),
        "predictions": predictions,
        "bow_prediction_frames": fold_runner.bow_prediction_frames,
        "htr_nuisance_prediction_frames": fold_runner.htr_nuisance_prediction_frames,
        "htr_effect_prediction_frames": fold_runner.htr_effect_prediction_frames,
        "ensemble_nuisance_prediction_frames": (fold_runner.ensemble_nuisance_prediction_frames),
        "htr_attention_rows": fold_runner.htr_attention_rows,
        "importance_rows": fold_runner.importance_rows,
        "embedding_evidence_rows": fold_runner.embedding_evidence_rows,
        "agent_rows": fold_runner.agent_rows,
        "extracted_feature_diagnostic_rows": (fold_runner.extracted_feature_diagnostic_rows),
        "candidate_signal_review_rows": fold_runner.candidate_signal_review_rows,
        "parsimony_review_rows": fold_runner.parsimony_review_rows,
        "feature_set_rows": fold_runner.feature_set_rows,
        "outer_metric_rows": fold_runner.outer_metric_rows,
    }


def _build_multi_model_inner_candidate_bundle_worker(
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    artifact_dir: Path,
    outer_fold: int,
    discovery_df: pd.DataFrame,
    inner_fold: int,
    fit_pos: np.ndarray,
    heldout_pos: np.ndarray,
    total_inner_folds: int,
    num_workers: int,
) -> Dict[str, Any]:
    worker = MultiModelAgenticForestRunner(
        dataset=dataset,
        config=config,
        output_path=(
            Path(artifact_dir)
            / f"outer_{int(outer_fold):03d}_candidate_inner_{int(inner_fold):03d}"
            / "predictions.parquet"
        ),
        num_workers=num_workers,
    )
    return worker._build_inner_consistency_candidate_bundle(
        outer_fold=outer_fold,
        discovery_df=discovery_df,
        inner_fold=inner_fold,
        fit_pos=fit_pos,
        heldout_pos=heldout_pos,
        total_inner_folds=total_inner_folds,
    )


def _normalize_texts(values: Sequence[Any]) -> List[str]:
    return [_normalize_text(value) for value in values]


def _split_is_honest(train_idx: np.ndarray, test_idx: np.ndarray) -> bool:
    train_ids = {int(idx) for idx in np.asarray(train_idx, dtype=int).tolist()}
    test_ids = {int(idx) for idx in np.asarray(test_idx, dtype=int).tolist()}
    return bool(test_ids) and train_ids.isdisjoint(test_ids)


def _non_htr_prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "model_family" not in frame.columns:
        return frame.copy()
    mask = frame["model_family"].fillna("").astype(str).str.lower() != "htr"
    return frame.loc[mask].copy()


def _candidate_role_decision(spec: ExplicitFeatureSpec) -> str:
    roles = set(str(role) for role in spec.roles)
    if {"confounder", "effect_modifier"}.issubset(roles):
        return "dual_role_confounder_and_effect_modifier"
    if "confounder" in roles:
        return "confounder"
    if "effect_modifier" in roles:
        return "effect_modifier"
    return "unspecified_role"


def _normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).translate(_DASH_TRANSLATION)
    text = text.replace("\u2265", ">=").replace("\u2264", "<=")
    return text.lower()


def _format_selected_feature_roles(specs: Sequence[ExplicitFeatureSpec]) -> str:
    return ",".join(f"{spec.name}[{'+'.join(_ordered_roles(spec.roles))}]" for spec in specs)


def _ordered_roles(roles: Sequence[str]) -> List[str]:
    role_set = {str(role) for role in roles}
    ordered = [role for role in ("confounder", "effect_modifier") if role in role_set]
    ordered.extend(sorted(role_set.difference(ordered)))
    return ordered or ["unspecified"]


def _candidate_consistency_threshold(
    fold_count: int,
    *,
    min_folds: int,
    min_fold_fraction: float,
) -> int:
    if fold_count <= 0:
        return 1
    return min(
        int(fold_count),
        max(
            1,
            int(min_folds),
            int(np.ceil(float(min_fold_fraction) * int(fold_count))),
        ),
    )


def _resolve_alias_name(name: str, alias_map: Dict[str, str]) -> str:
    current = str(name)
    seen = set()
    while current in alias_map and current not in seen:
        seen.add(current)
        current = alias_map[current]
    return current


def _merge_duplicate_proposals(
    proposals: Sequence[AgenticFeatureProposal],
) -> List[AgenticFeatureProposal]:
    merged: Dict[str, AgenticFeatureProposal] = {}
    for proposal in proposals:
        if proposal.action != "add":
            continue
        if proposal.name in merged:
            merged[proposal.name] = _merge_proposals(merged[proposal.name], proposal)
        else:
            merged[proposal.name] = proposal
    return list(merged.values())


def _fallback_consistency_proposals(
    candidate_summaries: Sequence[Dict[str, Any]],
    canonical_proposals: Dict[str, AgenticFeatureProposal],
) -> List[AgenticFeatureProposal]:
    selected = [
        canonical_proposals[item["name"]]
        for item in _rank_consistency_summaries(candidate_summaries)
        if item.get("passes_consistency_gate") and item.get("name") in canonical_proposals
    ]
    if selected:
        return selected
    full_supported = [
        canonical_proposals[item["name"]]
        for item in _rank_consistency_summaries(candidate_summaries)
        if item.get("proposed_on_full_outer_train") and item.get("name") in canonical_proposals
    ]
    return full_supported


def _require_complete_consistency_fallback(
    proposals: Sequence[AgenticFeatureProposal],
    *,
    max_selected: int,
) -> List[AgenticFeatureProposal]:
    """Reject a fallback that would need to discard otherwise eligible candidates."""

    complete = list(proposals)
    if len(complete) > int(max_selected):
        raise RuntimeError(
            f"deterministic consistency fallback produced {len(complete)} "
            f"eligible candidates, exceeding configured "
            f"candidate_proposals_per_fold={int(max_selected)}; refusing "
            "silent fallback-candidate omission"
        )
    return complete


def _agentic_consistency_selected_proposals(
    raw_selection: Any,
    *,
    candidate_summaries: Sequence[Dict[str, Any]],
    canonical_proposals: Dict[str, AgenticFeatureProposal],
    max_selected: int,
) -> Tuple[List[AgenticFeatureProposal], List[Dict[str, Any]]]:
    allowed_names = {
        _normalize_feature_name(item.get("name", ""))
        for item in candidate_summaries
        if isinstance(item, dict) and item.get("name")
    }
    raw_items = _raw_proposal_items(raw_selection)
    selected: List[AgenticFeatureProposal] = []
    rejected: List[Dict[str, Any]] = []
    selected_names = set()
    for raw in raw_items:
        if not isinstance(raw, dict):
            rejected.append({"proposal": raw, "reason": "proposal_not_object"})
            continue
        action = str(raw.get("action", "")).strip().lower()
        if action == "none":
            continue
        if action != "add":
            rejected.append({"proposal": raw, "reason": "invalid_selection_action"})
            continue
        name = _normalize_feature_name(raw.get("name", ""))
        if not name:
            rejected.append({"proposal": raw, "reason": "missing_name"})
            continue
        if name not in allowed_names:
            rejected.append({"proposal": raw, "reason": "name_not_in_candidate_summaries"})
            continue
        proposal = canonical_proposals.get(name)
        if proposal is None:
            rejected.append({"proposal": raw, "reason": "missing_canonical_proposal"})
            continue
        if name in selected_names:
            rejected.append({"proposal": raw, "reason": "duplicate_selection"})
            continue
        selected.append(proposal)
        selected_names.add(name)
    if len(selected) > int(max_selected):
        raise ValueError(
            f"agent consistency response selected {len(selected)} candidates, "
            f"exceeding configured max_selected={int(max_selected)}; refusing "
            "silent response truncation"
        )
    return selected, rejected


def _raw_proposal_items(raw: Any) -> List[Any]:
    if isinstance(raw, dict) and isinstance(raw.get("proposals"), list):
        return list(raw.get("proposals") or [])
    if isinstance(raw, list):
        return list(raw)
    return []


def _rank_consistency_summaries(
    candidate_summaries: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return sorted(
        candidate_summaries,
        key=lambda item: (
            -int(bool(item.get("passes_consistency_gate"))),
            -int(item.get("inner_support_count") or 0),
            -int(bool(item.get("proposed_on_full_outer_train"))),
            str(item.get("name") or ""),
        ),
    )


def _build_clustered_concept_inventory_payload(
    *,
    bow_context: Dict[str, Any],
    outer_fold: int,
    max_concepts: int,
    embedding_encoder: Any,
) -> Dict[str, Any]:
    units = _harvest_concept_evidence_units(
        bow_context=bow_context,
        outer_fold=outer_fold,
    )
    candidates = _aggregate_concept_units(units)
    ranked_candidates = _rank_concept_candidates(candidates)
    if not ranked_candidates:
        return {
            "generation": {
                "schema_version": _CONCEPT_INVENTORY_SCHEMA_VERSION,
                "n_evidence_units": int(len(units)),
                "n_candidate_phrases": 0,
                "n_raw_clusters": 0,
                "n_agent_clusters": 0,
                "skipped": "no_candidate_phrases",
            },
            "agent_clusters": [],
        }

    tfidf_clusters, tfidf_skip = _cluster_concept_candidates_tfidf_svd(
        ranked_candidates,
        max_concepts=max_concepts,
    )
    embedding_candidates = ranked_candidates[
        : min(len(ranked_candidates), max(150, int(max_concepts) * 5))
    ]
    embedding_clusters, embedding_skip = _cluster_concept_candidates_embedding(
        embedding_candidates,
        embedding_encoder=embedding_encoder,
        max_concepts=max_concepts,
    )
    fused = _fuse_concept_clusters([*tfidf_clusters, *embedding_clusters])
    ranked_clusters = _rank_concept_clusters(fused)
    limit = min(
        len(ranked_clusters),
        max(1, min(_CONCEPT_CLUSTER_MAX_AGENT_CLUSTERS, max(20, int(max_concepts) * 2))),
    )
    agent_clusters = [
        _concept_cluster_agent_artifact(cluster, rank=rank)
        for rank, cluster in enumerate(ranked_clusters[:limit], start=1)
    ]
    generation: Dict[str, Any] = {
        "schema_version": _CONCEPT_INVENTORY_SCHEMA_VERSION,
        "n_evidence_units": int(len(units)),
        "n_candidate_phrases": int(len(ranked_candidates)),
        "n_tfidf_svd_clusters": int(len(tfidf_clusters)),
        "n_embedding_clusters": int(len(embedding_clusters)),
        "n_raw_clusters": int(len(tfidf_clusters) + len(embedding_clusters)),
        "n_fused_clusters": int(len(fused)),
        "n_agent_clusters": int(len(agent_clusters)),
        "methods": ["tfidf_svd", "embedding"],
    }
    if tfidf_skip:
        generation["tfidf_svd_skipped"] = tfidf_skip
    if embedding_skip:
        generation["embedding_skipped"] = embedding_skip
    return {
        "generation": generation,
        "agent_clusters": agent_clusters,
    }


def _harvest_concept_evidence_units(
    *,
    bow_context: Dict[str, Any],
    outer_fold: int,
) -> List[Dict[str, Any]]:
    units: List[Dict[str, Any]] = []
    importance = bow_context.get("feature_importance") or {}
    for key in ("phrase_consensus", "phrase_features"):
        for item in importance.get(key) or []:
            if not isinstance(item, dict):
                continue
            phrase = item.get("feature") or item.get("phrase")
            if not phrase:
                continue
            signal = []
            if (item.get("mean_abs_confounder_score") or item.get("best_abs_confounder_score") or 0) > 0:
                signal.append("bow_confounder")
            if (item.get("mean_abs_effect_score") or item.get("best_abs_effect_score") or 0) > 0:
                signal.append("bow_effect")
            score = float(item.get("supporting_view_count") or 1)
            score += 20.0 * float(item.get("mean_abs_confounder_score") or 0.0)
            score += 20.0 * float(item.get("mean_abs_effect_score") or 0.0)
            _add_concept_unit(
                units,
                phrase=phrase,
                source_family="bow",
                signal="+".join(signal) or "bow",
                fold_key=outer_fold,
                score=score,
            )
    for view in importance.get("views") or []:
        if not isinstance(view, dict):
            continue
        view_name = str(view.get("view_name") or "")
        for key in [
            "phrase_features",
            "confounder_overlap",
            "treatment_positive",
            "treatment_negative",
            "outcome_positive",
            "outcome_negative",
            "effect_positive",
            "effect_negative",
            "r_pseudo_outcome_positive",
            "r_pseudo_outcome_negative",
            "top_features",
        ]:
            for item in view.get(key) or []:
                if not isinstance(item, dict):
                    continue
                phrase = item.get("feature") or item.get("term") or item.get("phrase")
                if not phrase:
                    continue
                raw_score = item.get("score", item.get("coef", item.get("importance", 0.0)))
                try:
                    score = 1.0 + abs(float(raw_score or 0.0))
                except (TypeError, ValueError):
                    score = 1.0
                _add_concept_unit(
                    units,
                    phrase=phrase,
                    source_family="bow",
                    signal=f"{view_name}:{key}" if view_name else key,
                    fold_key=outer_fold,
                    score=score,
                )

    documents = _collect_concept_inventory_documents(bow_context, outer_fold=outer_fold)
    units.extend(_mine_ngram_concept_units_from_documents(documents))
    return units


def _collect_concept_inventory_documents(
    bow_context: Dict[str, Any],
    *,
    outer_fold: int,
) -> List[Dict[str, Any]]:
    documents: List[Dict[str, Any]] = []
    embedding = bow_context.get("embedding_contrast_evidence") or {}
    for contrast in embedding.get("contrasts") or []:
        if not isinstance(contrast, dict):
            continue
        signal = str(contrast.get("name") or contrast.get("role_hint") or "embedding_contrast")
        for chunk_key in [
            "positive_aligned_chunks",
            "negative_aligned_chunks",
            "positive_external_chunks",
            "negative_external_chunks",
        ]:
            for chunk in contrast.get(chunk_key) or []:
                if not isinstance(chunk, dict):
                    continue
                text = chunk.get("text")
                if text:
                    documents.append(
                        {
                            "text": _concept_cluster_normalize_text(text),
                            "source_family": "embedding_contrast",
                            "signal": signal,
                            "fold_key": int(outer_fold),
                            "row_id": chunk.get("row_id"),
                        }
                    )
    htr = bow_context.get("htr_attention_evidence") or {}
    for section in ("nuisance", "effect"):
        attention = (htr.get(section) or {}).get("attention") or []
        for row in attention:
            if not isinstance(row, dict):
                continue
            text = row.get("evidence_snippet") or row.get("snippet")
            if text:
                documents.append(
                    {
                        "text": _concept_cluster_normalize_text(text),
                        "source_family": "htr",
                        "signal": str(row.get("target_source") or row.get("stage") or section),
                        "fold_key": int(outer_fold),
                        "row_id": row.get("row_id"),
                    }
                )
            spans = row.get("top_token_spans") or []
            for span in spans[:8] if isinstance(spans, list) else []:
                span_text = span.get("text") if isinstance(span, dict) else span
                if span_text:
                    documents.append(
                        {
                            "text": _concept_cluster_normalize_text(span_text),
                            "source_family": "htr",
                            "signal": str(row.get("target_source") or row.get("stage") or section),
                            "fold_key": int(outer_fold),
                            "row_id": row.get("row_id"),
                        }
                    )
    return [
        doc
        for doc in documents
        if len(str(doc.get("text") or "")) >= 8
    ]


def _mine_ngram_concept_units_from_documents(
    documents: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not documents:
        return []
    texts = [str(doc.get("text") or "") for doc in documents]
    min_df = 2 if len(texts) < 25 else 3
    try:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 4),
            min_df=min_df,
            max_df=0.65,
            stop_words=list(_concept_cluster_stop_words()),
            token_pattern=r"(?u)\b[a-z][a-z0-9%<>/=+\-]*\b|\b\d+(?:\.\d+)?%?\b",
            max_features=5000,
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return []
    terms = np.asarray(vectorizer.get_feature_names_out())
    units: List[Dict[str, Any]] = []
    for row_index, doc in enumerate(documents):
        row = matrix.getrow(row_index)
        if row.nnz == 0:
            continue
        top_positions = row.indices[np.argsort(row.data)[-8:]]
        for term_index in top_positions:
            phrase = str(terms[term_index])
            score = float(row[0, term_index]) * 3.0
            _add_concept_unit(
                units,
                phrase=phrase,
                source_family=str(doc.get("source_family") or "text"),
                signal=str(doc.get("signal") or ""),
                fold_key=doc.get("fold_key"),
                row_id=doc.get("row_id"),
                snippet=str(doc.get("text") or ""),
                score=score,
            )
    return units


def _add_concept_unit(
    units: List[Dict[str, Any]],
    *,
    phrase: Any,
    source_family: str,
    signal: str,
    fold_key: Any,
    row_id: Any = None,
    snippet: str = "",
    score: float = 1.0,
) -> None:
    normalized = _concept_cluster_normalize_phrase(phrase)
    if not _concept_cluster_phrase_ok(normalized):
        return
    units.append(
        {
            "phrase": normalized,
            "source_family": str(source_family or "unknown"),
            "signal": str(signal or ""),
            "fold_key": None if fold_key is None else int(fold_key),
            "row_id": None if row_id is None else str(row_id),
            "snippet": _concept_clip_text(snippet, _CONCEPT_CLUSTER_SNIPPET_CHARS),
            "score": float(score),
        }
    )


def _aggregate_concept_units(units: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    aggregated: Dict[str, Dict[str, Any]] = {}
    for unit in units:
        phrase = str(unit.get("phrase") or "")
        if not phrase:
            continue
        item = aggregated.setdefault(
            phrase,
            {
                "phrase": phrase,
                "source_counts": Counter(),
                "signal_counts": Counter(),
                "folds": set(),
                "rows": set(),
                "snippets": [],
                "score": 0.0,
            },
        )
        item["source_counts"][str(unit.get("source_family") or "unknown")] += 1
        signal = str(unit.get("signal") or "")
        if signal:
            item["signal_counts"][signal] += 1
        if unit.get("fold_key") is not None:
            item["folds"].add(int(unit["fold_key"]))
        if unit.get("row_id") is not None:
            item["rows"].add(str(unit["row_id"]))
        snippet = str(unit.get("snippet") or "")
        if snippet and len(item["snippets"]) < 6:
            item["snippets"].append(snippet)
        item["score"] += float(unit.get("score") or 0.0)
    candidates = []
    for item in aggregated.values():
        source_overlap = len(item["source_counts"])
        fold_count = len(item["folds"])
        bow_count = int(item["source_counts"].get("bow", 0))
        if source_overlap >= 2 or fold_count >= 2 or bow_count >= 2 or item["score"] >= 4.0:
            candidates.append(item)
    return candidates


def _rank_concept_candidates(candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        candidates,
        key=lambda item: (
            -len(item.get("source_counts") or {}),
            -len(item.get("folds") or []),
            -float(item.get("score") or 0.0),
            str(item.get("phrase") or ""),
        ),
    )


def _cluster_concept_candidates_tfidf_svd(
    candidates: Sequence[Dict[str, Any]],
    *,
    max_concepts: int,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    if not candidates:
        return [], "no_candidate_phrases"
    if len(candidates) <= 4:
        return [
            _make_concept_cluster("tfidf_svd", idx, [idx], candidates)
            for idx in range(len(candidates))
        ], None
    documents = [_concept_candidate_document(candidate) for candidate in candidates]
    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=1, sublinear_tf=True)
        matrix = vectorizer.fit_transform(documents)
    except ValueError as exc:
        return [], f"tfidf_failed:{exc}"
    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return [
            _make_concept_cluster("tfidf_svd", idx, [idx], candidates)
            for idx in range(len(candidates))
        ], None
    n_components = min(80, matrix.shape[0] - 1, matrix.shape[1] - 1)
    if n_components >= 2:
        vectors = TruncatedSVD(n_components=n_components, random_state=17).fit_transform(matrix)
    else:
        vectors = matrix.toarray()
    vectors = sklearn_normalize(vectors)
    n_clusters = _concept_cluster_count(len(candidates), max_concepts=max_concepts)
    if n_clusters >= len(candidates):
        return [
            _make_concept_cluster("tfidf_svd", idx, [idx], candidates)
            for idx in range(len(candidates))
        ], None
    labels = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=17,
        n_init=10,
        batch_size=max(64, min(512, len(candidates))),
    ).fit_predict(vectors)
    clusters = []
    for cluster_idx in range(n_clusters):
        indices = [idx for idx, label in enumerate(labels) if int(label) == cluster_idx]
        if indices:
            clusters.append(_make_concept_cluster("tfidf_svd", cluster_idx, indices, candidates))
    return clusters, None


def _cluster_concept_candidates_embedding(
    candidates: Sequence[Dict[str, Any]],
    *,
    embedding_encoder: Any,
    max_concepts: int,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    if len(candidates) < 4:
        return [], "too_few_candidate_phrases"
    documents = [_concept_candidate_document(candidate) for candidate in candidates]
    matrix, skip_reason = embedding_encoder(documents)
    if skip_reason:
        return [], skip_reason
    try:
        vectors = _coerce_concept_embedding_matrix(matrix, len(candidates))
    except Exception as exc:
        return [], f"invalid_embedding_matrix:{exc}"
    vectors = sklearn_normalize(vectors)
    n_clusters = _concept_cluster_count(len(candidates), max_concepts=max_concepts)
    if n_clusters >= len(candidates):
        return [
            _make_concept_cluster("embedding", idx, [idx], candidates)
            for idx in range(len(candidates))
        ], None
    labels = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=23,
        n_init=10,
        batch_size=max(64, min(512, len(candidates))),
    ).fit_predict(vectors)
    clusters = []
    for cluster_idx in range(n_clusters):
        indices = [idx for idx, label in enumerate(labels) if int(label) == cluster_idx]
        if indices:
            clusters.append(_make_concept_cluster("embedding", cluster_idx, indices, candidates))
    return clusters, None


def _concept_cluster_count(n_items: int, *, max_concepts: int) -> int:
    if n_items <= 0:
        return 0
    return max(
        2,
        min(
            int(n_items),
            max(4, int(np.sqrt(max(1, n_items)) * 2)),
            max(4, int(max_concepts) * 2),
            80,
        ),
    )


def _make_concept_cluster(
    method: str,
    cluster_idx: int,
    candidate_indices: Sequence[int],
    candidates: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    source_counts: Counter = Counter()
    signal_counts: Counter = Counter()
    folds = set()
    rows = set()
    score = 0.0
    phrases: List[str] = []
    snippets: List[str] = []
    for idx in candidate_indices:
        candidate = candidates[int(idx)]
        phrases.append(str(candidate.get("phrase") or ""))
        source_counts.update(candidate.get("source_counts") or {})
        signal_counts.update(candidate.get("signal_counts") or {})
        folds.update(candidate.get("folds") or set())
        rows.update(candidate.get("rows") or set())
        score += float(candidate.get("score") or 0.0)
        for snippet in candidate.get("snippets") or []:
            if snippet and len(snippets) < 6:
                snippets.append(_concept_clip_text(snippet, _CONCEPT_CLUSTER_SNIPPET_CHARS))
    top_phrases = _dedupe_phrase_list(
        sorted(
            set(phrases),
            key=lambda phrase: (
                -sum(1 for idx in candidate_indices if candidates[int(idx)].get("phrase") == phrase),
                -len(phrase.split()),
                phrase,
            ),
        ),
        limit=_CONCEPT_CLUSTER_TOP_PHRASES,
    )
    return {
        "cluster_id": f"{method}_{int(cluster_idx):03d}",
        "methods": [method],
        "candidate_indices": set(int(idx) for idx in candidate_indices),
        "source_counts": source_counts,
        "signal_counts": signal_counts,
        "folds": folds,
        "rows": rows,
        "score": float(score),
        "top_phrases": top_phrases,
        "exemplar_snippets": snippets[:4],
    }


def _fuse_concept_clusters(clusters: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    fused: List[Dict[str, Any]] = []
    for cluster in _rank_concept_clusters(clusters):
        phrase_set = set(cluster.get("top_phrases") or [])
        merged = False
        for existing in fused:
            existing_set = set(existing.get("top_phrases") or [])
            if _phrase_sets_should_merge(phrase_set, existing_set):
                _merge_concept_cluster(existing, cluster)
                merged = True
                break
        if not merged:
            fused.append(copy.deepcopy(cluster))
    return fused


def _phrase_sets_should_merge(left: set, right: set) -> bool:
    if not left or not right:
        return False
    if left & right:
        return True
    overlap = len(left & right) / max(1, min(len(left), len(right)))
    if overlap >= 0.35:
        return True
    for a in list(left)[:8]:
        for b in list(right)[:8]:
            if a and b and (a in b or b in a):
                return True
    return False


def _merge_concept_cluster(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    target["methods"] = sorted(set(target.get("methods") or []) | set(source.get("methods") or []))
    target["candidate_indices"] = set(target.get("candidate_indices") or set()) | set(
        source.get("candidate_indices") or set()
    )
    target["source_counts"].update(source.get("source_counts") or {})
    target["signal_counts"].update(source.get("signal_counts") or {})
    target["folds"] = set(target.get("folds") or set()) | set(source.get("folds") or set())
    target["rows"] = set(target.get("rows") or set()) | set(source.get("rows") or set())
    target["score"] = float(target.get("score") or 0.0) + float(source.get("score") or 0.0)
    target["top_phrases"] = _dedupe_phrase_list(
        list(target.get("top_phrases") or []) + list(source.get("top_phrases") or []),
        limit=_CONCEPT_CLUSTER_TOP_PHRASES,
    )
    target["exemplar_snippets"] = _dedupe_text_list(
        list(target.get("exemplar_snippets") or [])
        + list(source.get("exemplar_snippets") or []),
        limit=4,
    )


def _rank_concept_clusters(clusters: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        clusters,
        key=lambda cluster: (
            -len(cluster.get("source_counts") or {}),
            -len(cluster.get("folds") or []),
            -float(cluster.get("score") or 0.0),
            -len(cluster.get("candidate_indices") or []),
            str((cluster.get("top_phrases") or [""])[0]),
        ),
    )


def _concept_cluster_agent_artifact(cluster: Dict[str, Any], *, rank: int) -> Dict[str, Any]:
    cluster_id = f"cluster_{int(rank):03d}"
    source_counts = cluster.get("source_counts") or Counter()
    signal_counts = cluster.get("signal_counts") or Counter()
    return {
        "cluster_id": cluster_id,
        "methods": list(cluster.get("methods") or []),
        "source_families": [key for key, _ in Counter(source_counts).most_common()],
        "source_counts": dict(Counter(source_counts).most_common()),
        "source_overlap": int(len(source_counts)),
        "fold_count": int(len(cluster.get("folds") or [])),
        "row_count": int(len(cluster.get("rows") or [])),
        "signal_counts": dict(Counter(signal_counts).most_common(12)),
        "top_phrases": list(cluster.get("top_phrases") or [])[:_CONCEPT_CLUSTER_TOP_PHRASES],
        "example_values_or_phrases": list(cluster.get("top_phrases") or [])[:6],
        "exemplar_snippets": list(cluster.get("exemplar_snippets") or [])[:4],
        "score": round(float(cluster.get("score") or 0.0), 4),
    }


def _fallback_concepts_from_clusters(
    clusters: Sequence[Dict[str, Any]],
    *,
    max_concepts: int,
) -> List[Dict[str, Any]]:
    concepts: List[Dict[str, Any]] = []
    seen = set()
    for cluster in clusters:
        phrases = [str(phrase) for phrase in cluster.get("top_phrases") or [] if str(phrase)]
        if not phrases:
            continue
        label = phrases[0]
        name = _normalize_feature_name(label)
        if not name or name in seen:
            continue
        seen.add(name)
        concepts.append(
            {
                "name": name,
                "label": label,
                "value_kind": _infer_concept_value_kind_from_phrases(phrases),
                "source_families": cluster.get("source_families") or [],
                "source_overlap": int(cluster.get("source_overlap") or 0),
                "supporting_phrases": phrases[:6],
                "example_values_or_phrases": cluster.get("example_values_or_phrases") or phrases[:6],
                "extractability": _fallback_extractability(cluster),
                "cluster_ids": [cluster.get("cluster_id")],
                "notes": "Deterministic fallback label from clustered text evidence.",
            }
        )
        if len(concepts) >= int(max_concepts):
            break
    return concepts


def _merge_concept_inventory_concepts(
    concepts: Sequence[Dict[str, Any]],
    *,
    max_concepts: int,
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for raw in concepts:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or "").strip()
        if not name:
            label = str(raw.get("label") or "").strip()
            name = _normalize_feature_name(label)
        if not name:
            continue
        item = dict(raw)
        item["name"] = name
        existing = merged.get(name)
        if existing is None:
            merged[name] = item
            continue
        for key in (
            "source_families",
            "supporting_phrases",
            "example_values_or_phrases",
            "cluster_ids",
        ):
            existing[key] = _dedupe_any_list(
                list(existing.get(key) or []) + list(item.get(key) or []),
                limit=24 if key != "cluster_ids" else 120,
            )
        existing["source_overlap"] = max(
            int(existing.get("source_overlap") or 0),
            int(item.get("source_overlap") or 0),
            len(existing.get("source_families") or []),
        )
        if not existing.get("label") and item.get("label"):
            existing["label"] = item.get("label")
        if not existing.get("value_kind") and item.get("value_kind"):
            existing["value_kind"] = item.get("value_kind")
        existing["extractability"] = _best_extractability(
            existing.get("extractability"),
            item.get("extractability"),
        )
        if item.get("notes") and item.get("notes") not in str(existing.get("notes") or ""):
            existing["notes"] = _concept_clip_text(
                " ".join(
                    part
                    for part in [str(existing.get("notes") or ""), str(item.get("notes") or "")]
                    if part
                ),
                500,
            )
    return list(merged.values())[: max(1, int(max_concepts))]


def _dedupe_any_list(values: Sequence[Any], *, limit: int) -> List[Any]:
    result: List[Any] = []
    seen = set()
    for value in values:
        key = json.dumps(value, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
        if len(result) >= int(limit):
            break
    return result


def _best_extractability(left: Any, right: Any) -> str:
    rank = {"high": 3, "medium": 2, "low": 1}
    left_text = str(left or "").lower()
    right_text = str(right or "").lower()
    return left_text if rank.get(left_text, 0) >= rank.get(right_text, 0) else right_text


def _infer_concept_value_kind_from_phrases(phrases: Sequence[str]) -> str:
    joined = " ".join(phrases)
    if re.search(r"\d", joined):
        return "continuous"
    if len(set(phrases)) <= 3:
        return "categorical"
    return "unknown"


def _fallback_extractability(cluster: Dict[str, Any]) -> str:
    if int(cluster.get("source_overlap") or 0) >= 2:
        return "high"
    if int(cluster.get("fold_count") or 0) >= 2:
        return "medium"
    return "low"


def _concept_candidate_document(candidate: Dict[str, Any]) -> str:
    phrase = str(candidate.get("phrase") or "")
    sources = " ".join((candidate.get("source_counts") or Counter()).keys())
    signals = " ".join(key for key, _ in (candidate.get("signal_counts") or Counter()).most_common(8))
    snippets = " ".join(list(candidate.get("snippets") or [])[:3])
    return " ".join([phrase] * 8 + [sources, signals, snippets])


def _concept_cluster_normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).translate(_DASH_TRANSLATION)
    text = text.replace("\u2265", ">=").replace("\u2264", "<=")
    text = text.lower()
    text = re.sub(r"([a-z])[-_/]([a-z0-9])", r"\1 \2", text)
    text = re.sub(r"[^a-z0-9%\.<>/=+ -]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _concept_cluster_normalize_phrase(value: Any) -> str:
    text = _concept_cluster_normalize_text(value)
    text = re.sub(r"\s+", " ", text).strip(" -")
    return text


def _concept_cluster_phrase_ok(phrase: str) -> bool:
    if not phrase:
        return False
    tokens = phrase.split()
    if not tokens:
        return False
    if len(tokens) == 1 and (tokens[0] in _concept_cluster_stop_words() or len(tokens[0]) < 3):
        return False
    if all(token in _concept_cluster_stop_words() for token in tokens):
        return False
    if all(re.fullmatch(r"\d+(?:\.\d+)?%?", token) for token in tokens):
        return False
    numeric_count = sum(bool(re.fullmatch(r"\d+(?:\.\d+)?%?", token)) for token in tokens)
    if len(tokens) > 1 and numeric_count > max(1, len(tokens) // 2):
        return False
    return True


def _concept_cluster_stop_words() -> set:
    return set(ENGLISH_STOP_WORDS) | {
        "patient",
        "patients",
        "report",
        "reports",
        "date",
        "omitted",
        "specimen",
        "clinical",
        "history",
        "description",
        "gross",
        "microscopic",
        "sections",
        "section",
        "received",
        "formalin",
        "processed",
        "embedded",
        "paraffin",
        "stained",
        "additional",
        "core",
        "cores",
        "fragment",
        "fragments",
        "tissue",
        "tumor",
        "tumour",
        "cell",
        "cells",
        "lung",
        "mass",
        "right",
        "left",
        "upper",
        "lower",
        "lobe",
        "biopsy",
        "needle",
        "bronchoscopic",
        "diagnosis",
        "assessment",
        "plan",
        "table",
        "result",
        "interpretation",
        "study",
        "studies",
        "cm",
        "mm",
        "ml",
        "mg",
        "dl",
        "mrn",
        "id",
        "accession",
    }


def _dedupe_phrase_list(values: Sequence[str], *, limit: int) -> List[str]:
    result: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        if any(text == existing or text in existing or existing in text for existing in result):
            continue
        result.append(text)
        if len(result) >= int(limit):
            break
    return result


def _dedupe_text_list(values: Sequence[str], *, limit: int) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        text = _concept_clip_text(value, _CONCEPT_CLUSTER_SNIPPET_CHARS)
        key = text[:80]
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
        if len(result) >= int(limit):
            break
    return result


def _concept_clip_text(value: Any, limit: int) -> str:
    text = str(value or "")
    if len(text) <= int(limit):
        return text
    return text[: int(limit)].rstrip()


def _coerce_concept_embedding_matrix(embeddings: Any, expected_rows: int) -> np.ndarray:
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise ValueError(f"expected 2D embedding matrix, got shape={matrix.shape}")
    if int(matrix.shape[0]) != int(expected_rows):
        raise ValueError(
            f"expected {expected_rows} embedding rows, got {int(matrix.shape[0])}"
        )
    return matrix


def _concept_inventory_is_current(value: Any) -> bool:
    return isinstance(value, dict) and value.get("schema_version") == _CONCEPT_INVENTORY_SCHEMA_VERSION


def _agent_row_has_concept_inventory(
    row: Dict[str, Any],
    *,
    consistency_enabled: bool,
) -> bool:
    if not isinstance(row, dict):
        return False
    if not consistency_enabled:
        return _concept_inventory_is_current(row.get("concept_inventory"))
    bundles = row.get("proposal_bundles")
    if not isinstance(bundles, list) or not bundles:
        return False
    return all(
        isinstance(bundle, dict) and _concept_inventory_is_current(bundle.get("concept_inventory"))
        for bundle in bundles
    )


def _concept_inventory_artifact(inventory: Dict[str, Any]) -> Dict[str, Any]:
    return dict(inventory)


def _agent_visible_concept_inventory(
    inventory: Optional[Dict[str, Any]],
    *,
    max_concepts: int,
) -> Optional[Dict[str, Any]]:
    if not _concept_inventory_is_current(inventory):
        return None
    concepts = inventory.get("concepts")
    if not isinstance(concepts, list):
        response = inventory.get("response")
        if isinstance(response, dict):
            concepts = response.get("concepts")
    if not isinstance(concepts, list):
        return None

    visible: List[Dict[str, Any]] = []
    allowed_keys = [
        "name",
        "label",
        "value_kind",
        "source_families",
        "source_overlap",
        "supporting_phrases",
        "example_values_or_phrases",
        "extractability",
        "cluster_ids",
        "notes",
    ]
    for item in concepts[: max(1, int(max_concepts))]:
        if not isinstance(item, dict):
            continue
        compact = {
            key: item.get(key)
            for key in allowed_keys
            if item.get(key) not in (None, "", [])
        }
        if compact.get("name"):
            visible.append(compact)
    if not visible:
        return None
    return {
        "source": _CONCEPT_INVENTORY_SCHEMA_VERSION,
        "concepts": visible,
    }


def _proposal_bundle_artifact(bundle: Dict[str, Any]) -> Dict[str, Any]:
    artifact = {key: value for key, value in bundle.items() if key not in {"valid_proposals"}}
    artifact["valid_proposals"] = [
        _proposal_to_dict(proposal) for proposal in bundle.get("valid_proposals", [])
    ]
    return artifact


def _proposal_to_dict(proposal: AgenticFeatureProposal) -> Dict[str, Any]:
    return {
        "action": proposal.action,
        "name": proposal.name,
        "type": proposal.type,
        "categories": proposal.categories,
        "roles": proposal.roles,
        "description": proposal.description,
        "rationale": proposal.rationale,
        "expected_signal": proposal.expected_signal,
    }


def _proposal_from_dict(payload: Dict[str, Any]) -> AgenticFeatureProposal:
    return AgenticFeatureProposal(
        action=str(payload.get("action") or "add"),
        name=str(payload.get("name") or ""),
        type=payload.get("type"),
        categories=payload.get("categories"),
        roles=list(payload.get("roles") or []),
        description=payload.get("description"),
        rationale=payload.get("rationale"),
        expected_signal=payload.get("expected_signal"),
    )


def _feature_spec_from_dict(payload: Dict[str, Any]) -> ExplicitFeatureSpec:
    return ExplicitFeatureSpec(
        name=str(payload["name"]),
        type=str(payload.get("type") or "continuous"),
        categories=payload.get("categories"),
        description=payload.get("description"),
        roles=list(payload.get("roles") or []),
        value_aliases=payload.get("value_aliases"),
    )


def _columns_to_feature_dicts(
    df: pd.DataFrame,
    specs: Sequence[ExplicitFeatureSpec],
) -> Optional[List[Dict[str, Any]]]:
    if not specs:
        return None
    values: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        item: Dict[str, Any] = {}
        for spec in specs:
            value_col = f"explicit_feat_{spec.name}"
            legacy_col = f"explicit_conf_{spec.name}"
            source_col = value_col if value_col in df.columns else legacy_col
            value = row.get(source_col)
            missing_col = f"{source_col}_missing"
            item[spec.name] = value
            item[f"{spec.name}_missing"] = bool(row.get(missing_col, pd.isna(value)))
        values.append(item)
    return values


def _candidate_features_frame(dataset: pd.DataFrame) -> pd.DataFrame:
    id_cols = [column for column in ["_oci_row_id"] if column in dataset.columns]
    feature_cols = [column for column in dataset.columns if column.startswith("explicit_feat_")]
    cols = [*id_cols, *sorted(feature_cols)]
    if cols:
        return dataset[cols].copy()
    return pd.DataFrame({"_oci_row_id": np.arange(len(dataset), dtype=int)})


def _ensemble_nuisance_artifact_frame(
    *,
    source_frames: Sequence[pd.DataFrame],
    ensemble_frames: Sequence[pd.DataFrame],
) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for frame in source_frames:
        if frame is None or frame.empty or not {"e_hat", "m_hat"}.issubset(frame.columns):
            continue
        local = frame.copy()
        view_name = (
            local["view_name"].fillna("").astype(str)
            if "view_name" in local.columns
            else pd.Series([""] * len(local), index=local.index)
        )
        mask = ~view_name.str.startswith("ensemble_r__")
        mask &= view_name != "htr_effect"
        local = local.loc[mask].copy()
        if local.empty:
            continue
        local["nuisance_record_type"] = "source"
        local["source_name"] = _nuisance_source_name(local)
        if "r_pseudo_outcome" not in local.columns and "pseudo_target" in local.columns:
            local["r_pseudo_outcome"] = local["pseudo_target"]
        rows.append(local)
    for frame in ensemble_frames:
        if frame is None or frame.empty:
            continue
        local = frame.copy()
        local["nuisance_record_type"] = "ensemble_mean"
        local["source_name"] = local.get("target_source", "ensemble_mean_nuisance")
        if "pseudo_target" not in local.columns and "r_pseudo_outcome" in local.columns:
            local["pseudo_target"] = local["r_pseudo_outcome"]
        rows.append(local)
    if not rows:
        return pd.DataFrame()
    combined = pd.concat(rows, ignore_index=True, sort=False)
    preferred = [
        "_oci_row_id",
        "outer_fold",
        "nuisance_fold",
        "nuisance_record_type",
        "source_name",
        "view_name",
        "target_source",
        "e_hat",
        "m_hat",
        "y_residual",
        "t_residual",
        "pseudo_target",
        "r_pseudo_outcome",
        "r_loss",
        "r_loss_at_zero_tau",
    ]
    ordered = [column for column in preferred if column in combined.columns]
    ordered.extend([column for column in combined.columns if column not in ordered])
    return combined[ordered]


def _nuisance_source_name(frame: pd.DataFrame) -> pd.Series:
    if "view_name" in frame.columns:
        return frame["view_name"].fillna("").astype(str)
    if "target_source" in frame.columns:
        return frame["target_source"].fillna("").astype(str)
    return pd.Series(["source"] * len(frame), index=frame.index)


def _evaluate_extracted_feature_set_diagnostic(
    *,
    train_df: pd.DataFrame,
    specs: Sequence[ExplicitFeatureSpec],
    config: AppliedInferenceConfig,
    nn_config: MultiModelAgenticForestConfig,
    bow_metrics: Dict[str, Any],
    embedding_evidence: Dict[str, Any],
    random_state: int,
) -> Dict[str, Any]:
    y = train_df[config.outcome_column].to_numpy(dtype=float)
    t = train_df[config.treatment_column].to_numpy(dtype=float)
    specs = list(specs)
    extraction_summary = _summarize_multi_model_extractions(train_df, specs)
    x_full, x_names = _explicit_matrix_full(train_df, specs, role="effect_modifier")
    w_full, w_names = _explicit_matrix_full(train_df, specs, role="confounder")

    status = "ok"
    if not specs:
        status = "no_selected_features"
    elif x_full.shape[1] == 0 and w_full.shape[1] == 0:
        status = "no_usable_feature_columns"

    e_hat = _crossfit_explicit_binary(
        train_df=train_df,
        labels=t,
        specs=specs,
        role="confounder",
        requested_folds=int(nn_config.nuisance_folds),
        random_state=random_state + 11,
    )
    if str(config.outcome_type).lower() == "continuous":
        m_hat = _crossfit_explicit_regression(
            train_df=train_df,
            values=y,
            specs=specs,
            role="confounder",
            requested_folds=int(nn_config.nuisance_folds),
            sample_weight=None,
            random_state=random_state + 23,
        )
    else:
        m_hat = _crossfit_explicit_binary(
            train_df=train_df,
            labels=y,
            specs=specs,
            role="confounder",
            requested_folds=int(nn_config.nuisance_folds),
            random_state=random_state + 23,
        )

    e_clipped = np.clip(e_hat, float(nn_config.e_clip), 1.0 - float(nn_config.e_clip))
    t_resid = t - e_clipped
    y_resid = y - m_hat
    pseudo_target = y_resid / t_resid
    pseudo_weight = np.square(t_resid)
    tau_hat = _crossfit_explicit_regression(
        train_df=train_df,
        values=pseudo_target,
        specs=specs,
        role="effect_modifier",
        requested_folds=int(nn_config.effect_folds),
        sample_weight=pseudo_weight,
        random_state=random_state + 37,
    )
    r_loss = np.square(y_resid - tau_hat * t_resid)
    r_loss_at_zero = np.square(y_resid)

    metrics: Dict[str, Any] = {
        "status": status,
        "n_rows": int(len(train_df)),
        "n_selected_features": int(len(specs)),
        "n_w_features": int(w_full.shape[1]),
        "n_x_features": int(x_full.shape[1]),
        "w_feature_names": w_names,
        "x_feature_names": x_names,
        "treatment_auroc": _safe_roc_auc(t, e_hat),
        "treatment_brier": _safe_brier_score(t, e_hat),
        "treatment_log_loss": _safe_log_loss(t, e_hat),
        "pseudo_target_mean": _finite_or_none(np.mean(pseudo_target)),
        "pseudo_target_std": _finite_or_none(np.std(pseudo_target)),
        "tau_hat_mean": _finite_or_none(np.mean(tau_hat)),
        "tau_hat_std": _finite_or_none(np.std(tau_hat)),
        "tau_hat_pseudo_target_corr": _safe_corr(tau_hat, pseudo_target),
        "r_loss_mean": _finite_or_none(np.mean(r_loss)),
        "r_loss_at_zero_tau_mean": _finite_or_none(np.mean(r_loss_at_zero)),
    }
    zero = metrics["r_loss_at_zero_tau_mean"]
    loss = metrics["r_loss_mean"]
    if zero is not None and zero > 0.0 and loss is not None:
        metrics["r_loss_relative_improvement"] = float(1.0 - loss / zero)
    if str(config.outcome_type).lower() == "continuous":
        metrics["outcome_rmse"] = _finite_or_none(np.sqrt(mean_squared_error(y, m_hat)))
    else:
        metrics["outcome_auroc"] = _safe_roc_auc(y, m_hat)
        metrics["outcome_brier"] = _safe_brier_score(y, m_hat)
        metrics["outcome_log_loss"] = _safe_log_loss(y, m_hat)

    return {
        "metrics": metrics,
        "benchmark": _extracted_feature_review_benchmarks(
            bow_metrics,
            embedding_evidence,
        ),
        "extraction_summary": extraction_summary,
    }


def _crossfit_explicit_binary(
    *,
    train_df: pd.DataFrame,
    labels: np.ndarray,
    specs: Sequence[ExplicitFeatureSpec],
    role: Optional[str],
    requested_folds: int,
    random_state: int,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=float)
    oof = np.full(len(labels), np.nan, dtype=float)
    if len(labels) == 0:
        return oof
    if len(np.unique(labels.astype(int))) < 2:
        return np.full(len(labels), float(np.nanmean(labels)), dtype=float)
    try:
        split_items = _binary_split_items(
            labels.astype(int),
            requested_folds=requested_folds,
            random_state=random_state,
        )
    except ValueError:
        return np.full(len(labels), float(np.nanmean(labels)), dtype=float)

    for fold, (fit_pos, heldout_pos) in enumerate(split_items, start=1):
        del fold
        fit_pos = np.asarray(fit_pos)
        heldout_pos = np.asarray(heldout_pos)
        fit_y = labels[fit_pos].astype(int)
        if len(np.unique(fit_y)) < 2:
            oof[heldout_pos] = float(np.mean(fit_y))
            continue
        x_fit, x_heldout = _explicit_matrix_split(
            train_df=train_df,
            fit_pos=fit_pos,
            heldout_pos=heldout_pos,
            specs=specs,
            role=role,
        )
        x_fit = _ensure_model_matrix(x_fit)
        x_heldout = _ensure_model_matrix(x_heldout)
        model = LogisticRegression(
            C=1.0,
            solver="liblinear",
            max_iter=1000,
            random_state=random_state,
        )
        try:
            model.fit(x_fit, fit_y)
            oof[heldout_pos] = model.predict_proba(x_heldout)[:, 1]
        except ValueError:
            oof[heldout_pos] = float(np.mean(fit_y))
    return _fill_nonfinite_predictions(oof, labels)


def _crossfit_explicit_regression(
    *,
    train_df: pd.DataFrame,
    values: np.ndarray,
    specs: Sequence[ExplicitFeatureSpec],
    role: Optional[str],
    requested_folds: int,
    sample_weight: Optional[np.ndarray],
    random_state: int,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    oof = np.full(len(values), np.nan, dtype=float)
    if len(values) == 0:
        return oof
    try:
        folds = _bounded_fold_count(requested_folds, len(values))
    except ValueError:
        return np.full(len(values), float(np.nanmean(values)), dtype=float)
    splitter = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    for fit_pos, heldout_pos in splitter.split(train_df):
        fit_pos = np.asarray(fit_pos)
        heldout_pos = np.asarray(heldout_pos)
        x_fit, x_heldout = _explicit_matrix_split(
            train_df=train_df,
            fit_pos=fit_pos,
            heldout_pos=heldout_pos,
            specs=specs,
            role=role,
        )
        x_fit = _ensure_model_matrix(x_fit)
        x_heldout = _ensure_model_matrix(x_heldout)
        model = Ridge(alpha=1.0, random_state=random_state)
        fit_weight = None
        if weights is not None and len(weights) == len(values):
            fit_weight = weights[fit_pos]
            fit_weight = np.where(
                np.isfinite(fit_weight) & (fit_weight > 0.0),
                fit_weight,
                0.0,
            )
            if float(np.sum(fit_weight)) <= 0.0:
                fit_weight = None
        finite = np.isfinite(values[fit_pos])
        if np.sum(finite) < 1:
            oof[heldout_pos] = float(np.nanmean(values))
            continue
        _fit_regressor(
            model,
            x_fit[finite],
            values[fit_pos][finite],
            sample_weight=None if fit_weight is None else fit_weight[finite],
            unsupported_sample_weight_policy="fail_closed",
        )
        oof[heldout_pos] = model.predict(x_heldout)
    return _fill_nonfinite_predictions(oof, values)


def _explicit_matrix_split(
    *,
    train_df: pd.DataFrame,
    fit_pos: np.ndarray,
    heldout_pos: np.ndarray,
    specs: Sequence[ExplicitFeatureSpec],
    role: Optional[str],
) -> Tuple[np.ndarray, np.ndarray]:
    fit_df = train_df.iloc[np.asarray(fit_pos)]
    heldout_df = train_df.iloc[np.asarray(heldout_pos)]
    fit_dicts = _columns_to_feature_dicts(fit_df, specs) or []
    heldout_dicts = _columns_to_feature_dicts(heldout_df, specs) or []
    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}
    fit_features, _ = get_raw_explicit_features(
        fit_dicts,
        list(specs),
        continuous_means=means,
        continuous_stds=stds,
        role=role,
    )
    heldout_features, _ = get_raw_explicit_features(
        heldout_dicts,
        list(specs),
        continuous_means=means,
        continuous_stds=stds,
        role=role,
    )
    return (
        _as_2d_feature_matrix(fit_features, len(fit_df)),
        _as_2d_feature_matrix(heldout_features, len(heldout_df)),
    )


def _explicit_matrix_full(
    df: pd.DataFrame,
    specs: Sequence[ExplicitFeatureSpec],
    *,
    role: Optional[str],
) -> Tuple[np.ndarray, List[str]]:
    feature_dicts = _columns_to_feature_dicts(df, specs) or []
    features, names = get_raw_explicit_features(
        feature_dicts,
        list(specs),
        continuous_means={},
        continuous_stds={},
        role=role,
    )
    return _as_2d_feature_matrix(features, len(df)), list(names)


def _as_2d_feature_matrix(values: Sequence[Sequence[float]], n_rows: int) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2:
        return np.zeros((n_rows, 0), dtype=np.float32)
    if matrix.shape[0] != n_rows:
        return np.zeros((n_rows, 0), dtype=np.float32)
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)


def _ensure_model_matrix(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        return np.zeros((matrix.shape[0], 1), dtype=np.float32)
    return matrix


def _fill_nonfinite_predictions(pred: np.ndarray, fallback_values: np.ndarray) -> np.ndarray:
    filled = np.asarray(pred, dtype=float).copy()
    finite = np.isfinite(filled)
    if np.all(finite):
        return filled
    fallback = float(np.nanmean(fallback_values)) if len(fallback_values) else 0.0
    if not np.isfinite(fallback):
        fallback = 0.0
    filled[~finite] = fallback
    return filled


def _safe_log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    try:
        return _finite_or_none(
            log_loss(
                np.asarray(y_true, dtype=int),
                np.clip(np.asarray(y_pred, dtype=float), 1e-6, 1.0 - 1e-6),
                labels=[0, 1],
            )
        )
    except ValueError:
        return None


def _safe_brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    try:
        return _finite_or_none(
            brier_score_loss(
                np.asarray(y_true, dtype=int),
                np.clip(np.asarray(y_pred, dtype=float), 0.0, 1.0),
            )
        )
    except ValueError:
        return None


def _align_htr_prediction_frame(
    frame: Any,
    discovery_df: pd.DataFrame,
    *,
    required_columns: Sequence[str],
    source: str,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"{source} did not return a predictions DataFrame")
    if "_oci_row_id" not in frame.columns:
        raise ValueError(f"{source} predictions must include _oci_row_id")
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} predictions missing required columns: {missing}")
    if frame["_oci_row_id"].duplicated().any():
        raise ValueError(f"{source} predictions contain duplicate _oci_row_id values")

    aligned = discovery_df[["_oci_row_id"]].merge(
        frame.copy(),
        on="_oci_row_id",
        how="left",
        sort=False,
    )
    if len(aligned) != len(discovery_df):
        raise ValueError(f"{source} predictions could not be aligned to discovery rows")
    for column in required_columns:
        values = pd.to_numeric(aligned[column], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{source} predictions contain non-finite {column} values")
        aligned[column] = values
    return aligned


def _htr_nuisance_metrics(
    *,
    discovery_df: pd.DataFrame,
    predictions: pd.DataFrame,
    treatment_column: str,
    outcome_column: str,
    outcome_type: str,
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {"n_rows": int(len(predictions))}
    e_hat = predictions["e_hat"].to_numpy(dtype=float)
    m_hat = predictions["m_hat"].to_numpy(dtype=float)
    metrics.update(
        {
            "e_hat_mean": _finite_or_none(np.mean(e_hat)),
            "e_hat_std": _finite_or_none(np.std(e_hat)),
            "m_hat_mean": _finite_or_none(np.mean(m_hat)),
            "m_hat_std": _finite_or_none(np.std(m_hat)),
        }
    )
    if treatment_column in discovery_df.columns:
        t = discovery_df[treatment_column].to_numpy(dtype=float)
        metrics.update(
            {
                "treatment_auroc": _safe_roc_auc(t, e_hat),
                "treatment_brier": _safe_brier_score(t, e_hat),
                "treatment_log_loss": _safe_log_loss(t, e_hat),
            }
        )
    if outcome_column in discovery_df.columns:
        y = discovery_df[outcome_column].to_numpy(dtype=float)
        if str(outcome_type).lower() == "continuous":
            metrics["outcome_rmse"] = _finite_or_none(np.sqrt(mean_squared_error(y, m_hat)))
        else:
            metrics.update(
                {
                    "outcome_auroc": _safe_roc_auc(y, m_hat),
                    "outcome_brier": _safe_brier_score(y, m_hat),
                    "outcome_log_loss": _safe_log_loss(y, m_hat),
                }
            )
    for column in ["y_residual", "t_residual", "r_pseudo_outcome"]:
        if column in predictions.columns:
            values = pd.to_numeric(predictions[column], errors="coerce").to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            metrics[f"{column}_mean"] = _finite_or_none(np.mean(finite)) if len(finite) else None
            metrics[f"{column}_std"] = _finite_or_none(np.std(finite)) if len(finite) else None
    return metrics


def _htr_effect_metrics(predictions: pd.DataFrame) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {"n_rows": int(len(predictions))}
    for column in [
        "tau_hat_r_stage",
        "tau_logit_modifier",
        "r_pseudo_outcome",
        "r_loss",
        "effect_loss",
        "effect_loss_at_zero_tau",
    ]:
        if column not in predictions.columns:
            continue
        values = pd.to_numeric(predictions[column], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        metrics[f"{column}_mean"] = _finite_or_none(np.mean(finite)) if len(finite) else None
        metrics[f"{column}_std"] = _finite_or_none(np.std(finite)) if len(finite) else None
    loss = metrics.get("r_loss_mean")
    zero = metrics.get("effect_loss_at_zero_tau_mean")
    if zero is not None and zero > 0.0 and loss is not None:
        metrics["r_loss_relative_improvement"] = float(1.0 - loss / zero)
    if "effect_objective" in predictions.columns:
        objectives = sorted(
            {str(value) for value in predictions["effect_objective"].dropna() if str(value)}
        )
        if objectives:
            metrics["effect_objectives"] = objectives
    if "target_source" in predictions.columns:
        target_sources = sorted(
            {str(value) for value in predictions["target_source"].dropna() if str(value)}
        )
        if target_sources:
            metrics["target_sources"] = target_sources
    return metrics


def _summarize_multi_model_extractions(
    df: pd.DataFrame,
    specs: Sequence[ExplicitFeatureSpec],
) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for spec in specs:
        value_col = f"explicit_feat_{spec.name}"
        missing_col = f"{value_col}_missing"
        if value_col not in df.columns:
            summaries.append(
                {
                    "name": spec.name,
                    "roles": list(spec.roles),
                    "coverage": 0.0,
                    "top_values": {},
                }
            )
            continue
        if missing_col in df.columns:
            missing = df[missing_col].astype(bool)
        else:
            missing = df[value_col].isna()
        observed = df.loc[~missing, value_col]
        summaries.append(
            {
                "name": spec.name,
                "roles": list(spec.roles),
                "coverage": float(1.0 - missing.mean()),
                "n_unique_observed": int(observed.nunique(dropna=True)),
                "top_values": observed.astype(str).value_counts().head(8).to_dict(),
            }
        )
    return summaries


def _feature_redundancy_review(
    *,
    train_df: pd.DataFrame,
    specs: Sequence[ExplicitFeatureSpec],
    corr_threshold: float,
) -> Dict[str, Any]:
    specs = list(specs)
    continuous_correlations: List[Dict[str, Any]] = []
    categorical_contingency: List[Dict[str, Any]] = []
    missingness_overlap: List[Dict[str, Any]] = []
    for left, right in combinations(specs, 2):
        left_values = _explicit_feature_series(train_df, left)
        right_values = _explicit_feature_series(train_df, right)
        left_missing = _explicit_feature_missing_mask(train_df, left, left_values)
        right_missing = _explicit_feature_missing_mask(train_df, right, right_values)
        missingness_overlap.append(
            {
                "a": left.name,
                "b": right.name,
                "both_missing": float(np.mean(left_missing & right_missing)),
                "either_missing": float(np.mean(left_missing | right_missing)),
            }
        )
        if left.type == "continuous" and right.type == "continuous":
            x = pd.to_numeric(left_values, errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(right_values, errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            corr = None
            if int(np.sum(mask)) >= 3 and np.std(x[mask]) > 0.0 and np.std(y[mask]) > 0.0:
                corr = _finite_or_none(np.corrcoef(x[mask], y[mask])[0, 1])
            if corr is not None and abs(corr) >= float(corr_threshold):
                continuous_correlations.append(
                    {
                        "a": left.name,
                        "b": right.name,
                        "correlation": float(corr),
                        "n_pairwise_complete": int(np.sum(mask)),
                    }
                )
        elif left.type == "categorical" and right.type == "categorical":
            left_cat = left_values.astype("object").where(~left_missing, "__MISSING__")
            right_cat = right_values.astype("object").where(~right_missing, "__MISSING__")
            table = pd.crosstab(left_cat, right_cat, dropna=False)
            total = float(table.to_numpy().sum())
            max_cell_fraction = None if total <= 0 else float(table.to_numpy().max() / total)
            categorical_contingency.append(
                {
                    "a": left.name,
                    "b": right.name,
                    "shape": [int(table.shape[0]), int(table.shape[1])],
                    "max_cell_fraction": max_cell_fraction,
                }
            )
    return {
        "continuous_correlations_abs_ge_threshold": continuous_correlations,
        "categorical_contingency": categorical_contingency,
        "missingness_overlap": missingness_overlap,
        "corr_threshold": float(corr_threshold),
    }


def _parsimony_feature_contract_document(spec: ExplicitFeatureSpec) -> str:
    categories = " ".join(str(value) for value in (spec.categories or []))
    aliases = " ".join(
        [
            str(value)
            for values in (getattr(spec, "value_aliases", None) or {}).values()
            for value in values
        ]
    )
    return " ".join(
        part
        for part in [
            spec.name.replace("_", " "),
            str(spec.description or ""),
            spec.type,
            categories,
            aliases,
            " ".join(spec.roles),
        ]
        if part
    )


def _parsimony_tfidf_semantic_vectors(documents: Sequence[str]) -> np.ndarray:
    documents = [str(value or "") for value in documents]
    n_items = len(documents)
    if n_items == 0:
        return np.zeros((0, 1), dtype=float)
    if n_items == 1:
        return np.ones((1, 1), dtype=float)
    try:
        word_matrix = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        ).fit_transform(documents)
        char_matrix = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            sublinear_tf=True,
        ).fit_transform(documents)
        matrix = sparse.hstack([word_matrix, char_matrix], format="csr")
    except ValueError:
        return np.eye(n_items, dtype=float)
    if matrix.shape[0] > 1 and all(
        (matrix[index] - matrix[0]).nnz == 0
        for index in range(1, matrix.shape[0])
    ):
        return np.ones((n_items, 1), dtype=float)
    n_components = min(64, matrix.shape[0] - 1, matrix.shape[1] - 1)
    if n_components >= 2:
        vectors = TruncatedSVD(
            n_components=n_components,
            random_state=97,
        ).fit_transform(matrix)
    else:
        vectors = matrix.toarray()
    return sklearn_normalize(np.asarray(vectors, dtype=float))


def _parsimony_feature_value_block(
    train_df: pd.DataFrame,
    spec: ExplicitFeatureSpec,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    values = _explicit_feature_series(train_df, spec)
    missing = _explicit_feature_missing_mask(train_df, spec, values)
    n_rows = len(train_df)
    if spec.type == "continuous":
        numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float).copy()
        numeric[missing] = np.nan
        observed = numeric[np.isfinite(numeric)]
        if len(observed):
            lower, upper = np.nanquantile(observed, [0.01, 0.99])
            clipped = np.clip(numeric, lower, upper)
            median = float(np.nanmedian(clipped))
            filled = np.where(np.isfinite(clipped), clipped, median)
            ranked = pd.Series(filled).rank(method="average", pct=True).to_numpy(dtype=float)
            std = float(np.std(ranked))
            standardized = (ranked - float(np.mean(ranked))) / (std if std > 0 else 1.0)
        else:
            standardized = np.zeros(n_rows, dtype=float)
        block = np.column_stack([standardized, missing.astype(float)])
        unique = int(pd.Series(observed).nunique(dropna=True)) if len(observed) else 0
    else:
        categorical = values.astype("object").where(~missing, "__MISSING__").astype(str)
        counts = categorical.value_counts(dropna=False)
        if len(counts) > 32:
            keep = set(counts.head(31).index.astype(str))
            categorical = categorical.where(categorical.isin(keep), "__OTHER__")
        block = pd.get_dummies(categorical, dtype=float).to_numpy(dtype=float)
        if block.shape[1] == 0:
            block = np.zeros((n_rows, 1), dtype=float)
        unique = int(values.loc[~missing].astype(str).nunique(dropna=True))

    block = np.array(block, dtype=float, copy=True)
    block -= np.mean(block, axis=0, keepdims=True)
    norms = np.linalg.norm(block, axis=0)
    norms = np.where(norms > 0.0, norms, 1.0)
    block = block / norms
    return block, {
        "coverage": float(1.0 - np.mean(missing)) if n_rows else 0.0,
        "n_unique_observed": unique,
        "encoded_columns": int(block.shape[1]),
    }


def _parsimony_empirical_sketch_matrix(
    *,
    train_df: pd.DataFrame,
    specs: Sequence[ExplicitFeatureSpec],
    sketch_dim: int,
    random_state: int,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    n_rows = len(train_df)
    if not specs:
        return np.zeros((0, max(1, sketch_dim + 3)), dtype=float), []
    rng = np.random.default_rng(int(random_state))
    projection = rng.normal(size=(max(1, n_rows), max(1, int(sketch_dim))))
    projection = sklearn_normalize(projection, axis=0)
    rows: List[np.ndarray] = []
    summaries: List[Dict[str, Any]] = []
    for spec in specs:
        block, summary = _parsimony_feature_value_block(train_df, spec)
        if n_rows:
            projected = projection[:n_rows].T @ block
            sketch = np.linalg.norm(projected, axis=1)
        else:
            sketch = np.zeros(max(1, int(sketch_dim)), dtype=float)
        rows.append(
            np.concatenate(
                [
                    sketch,
                    np.asarray(
                        [
                            float(summary["coverage"]),
                            min(1.0, np.log1p(summary["n_unique_observed"]) / np.log(33.0)),
                            1.0 if spec.type == "continuous" else 0.0,
                        ],
                        dtype=float,
                    ),
                ]
            )
        )
        summaries.append({"name": spec.name, **summary})
    matrix = np.vstack(rows)
    return sklearn_normalize(matrix), summaries


def _parsimony_mutual_neighbor_pairs(
    vectors: np.ndarray,
    *,
    neighbors: int,
) -> set:
    vectors = np.asarray(vectors, dtype=float)
    n_items = int(vectors.shape[0]) if vectors.ndim == 2 else 0
    if n_items < 2:
        return set()
    count = min(max(1, int(neighbors)), n_items - 1)
    model = NearestNeighbors(
        n_neighbors=count + 1,
        metric="cosine",
        algorithm="brute",
    ).fit(vectors)
    indices = model.kneighbors(vectors, return_distance=False)
    directed = {
        (row, int(col))
        for row, cols in enumerate(indices)
        for col in cols
        if int(col) != row
    }
    return {
        tuple(sorted((left, right)))
        for left, right in directed
        if (right, left) in directed and left != right
    }


def _parsimony_cramers_v(left: pd.Series, right: pd.Series) -> float:
    table = pd.crosstab(left, right, dropna=False)
    n = float(table.to_numpy().sum())
    if n <= 1.0 or min(table.shape) < 2:
        return 0.0
    try:
        chi2 = float(chi2_contingency(table, correction=False)[0])
    except ValueError:
        return 0.0
    rows, cols = table.shape
    phi2 = chi2 / n
    correction = ((cols - 1) * (rows - 1)) / max(1.0, n - 1.0)
    phi2_corrected = max(0.0, phi2 - correction)
    rows_corrected = rows - ((rows - 1) ** 2) / max(1.0, n - 1.0)
    cols_corrected = cols - ((cols - 1) ** 2) / max(1.0, n - 1.0)
    denominator = min(rows_corrected - 1.0, cols_corrected - 1.0)
    if denominator <= 0.0:
        return 0.0
    return float(np.clip(np.sqrt(phi2_corrected / denominator), 0.0, 1.0))


def _parsimony_correlation_ratio(
    continuous: np.ndarray,
    categorical: pd.Series,
) -> float:
    continuous = np.asarray(continuous, dtype=float)
    if len(continuous) < 3 or float(np.std(continuous)) <= 0.0:
        return 0.0
    grand_mean = float(np.mean(continuous))
    denominator = float(np.sum(np.square(continuous - grand_mean)))
    if denominator <= 0.0:
        return 0.0
    numerator = 0.0
    categories = categorical.astype(str).to_numpy()
    for value in np.unique(categories):
        group = continuous[categories == value]
        if len(group):
            numerator += float(len(group)) * float(np.mean(group) - grand_mean) ** 2
    return float(np.clip(np.sqrt(max(0.0, numerator / denominator)), 0.0, 1.0))


def _parsimony_pair_association(
    *,
    train_df: pd.DataFrame,
    left: ExplicitFeatureSpec,
    right: ExplicitFeatureSpec,
    missingness_weight: float,
) -> Dict[str, Any]:
    left_values = _explicit_feature_series(train_df, left)
    right_values = _explicit_feature_series(train_df, right)
    left_missing = _explicit_feature_missing_mask(train_df, left, left_values)
    right_missing = _explicit_feature_missing_mask(train_df, right, right_values)
    complete = ~(left_missing | right_missing)
    n_complete = int(np.sum(complete))
    association_type = f"{left.type}_{right.type}"
    value_association = 0.0
    if n_complete >= 3:
        if left.type == "continuous" and right.type == "continuous":
            x = pd.to_numeric(left_values[complete], errors="coerce")
            y = pd.to_numeric(right_values[complete], errors="coerce")
            valid = x.notna() & y.notna()
            corr = x[valid].corr(y[valid], method="spearman") if int(valid.sum()) >= 3 else None
            value_association = 0.0 if corr is None or not np.isfinite(corr) else abs(float(corr))
            association_type = "absolute_spearman"
        elif left.type == "categorical" and right.type == "categorical":
            value_association = _parsimony_cramers_v(
                left_values[complete].astype(str),
                right_values[complete].astype(str),
            )
            association_type = "bias_corrected_cramers_v"
        else:
            if left.type == "continuous":
                continuous = pd.to_numeric(left_values[complete], errors="coerce")
                categorical = right_values[complete].astype(str)
            else:
                continuous = pd.to_numeric(right_values[complete], errors="coerce")
                categorical = left_values[complete].astype(str)
            valid = continuous.notna()
            value_association = _parsimony_correlation_ratio(
                continuous[valid].to_numpy(dtype=float),
                categorical[valid],
            )
            association_type = "correlation_ratio"

    missing_association = 0.0
    if len(left_missing) >= 3 and np.std(left_missing) > 0 and np.std(right_missing) > 0:
        missing_corr = np.corrcoef(left_missing.astype(float), right_missing.astype(float))[0, 1]
        if np.isfinite(missing_corr):
            missing_association = abs(float(missing_corr))
    weight = float(np.clip(missingness_weight, 0.0, 1.0))
    empirical = (1.0 - weight) * value_association + weight * missing_association
    return {
        "a": left.name,
        "b": right.name,
        "association_type": association_type,
        "value_association": float(value_association),
        "missingness_association": float(missing_association),
        "empirical_similarity": float(np.clip(empirical, 0.0, 1.0)),
        "n_pairwise_complete": n_complete,
    }


def _parsimony_split_large_component(
    indices: Sequence[int],
    *,
    vectors: np.ndarray,
    max_size: int,
    random_state: int,
) -> List[List[int]]:
    indices = list(indices)
    if len(indices) <= max_size:
        return [indices]
    n_clusters = int(np.ceil(len(indices) / max(1, max_size)))
    labels = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=int(random_state),
        n_init=10,
        batch_size=max(32, len(indices)),
    ).fit_predict(vectors[np.asarray(indices)])
    unique_labels = sorted(set(int(value) for value in labels))
    if len(unique_labels) <= 1:
        return [
            indices[start : start + max_size]
            for start in range(0, len(indices), max_size)
        ]
    result: List[List[int]] = []
    for label in unique_labels:
        members = [indices[pos] for pos, value in enumerate(labels) if int(value) == label]
        result.extend(
            _parsimony_split_large_component(
                members,
                vectors=vectors,
                max_size=max_size,
                random_state=random_state + label + 1,
            )
        )
    return result


def _build_value_driven_feature_clusters(
    *,
    train_df: pd.DataFrame,
    specs: Sequence[ExplicitFeatureSpec],
    semantic_vectors: np.ndarray,
    nn_config: MultiModelAgenticForestConfig,
    random_state: int,
) -> Dict[str, Any]:
    specs = list(specs)
    n_specs = len(specs)
    if n_specs < 2:
        return {
            "generation": {
                "uses_actual_extracted_values": True,
                "n_features": n_specs,
                "skip_reason": "fewer_than_two_features",
            },
            "clusters": [],
        }
    empirical_vectors, value_summaries = _parsimony_empirical_sketch_matrix(
        train_df=train_df,
        specs=specs,
        sketch_dim=int(getattr(nn_config, "parsimony_cluster_sketch_dim", 32)),
        random_state=random_state,
    )
    semantic_vectors = sklearn_normalize(np.asarray(semantic_vectors, dtype=float))
    if semantic_vectors.shape[0] != n_specs:
        semantic_vectors = np.eye(n_specs, dtype=float)
    weight = float(getattr(nn_config, "parsimony_cluster_semantic_weight", 0.5))
    combined_vectors = sklearn_normalize(
        np.hstack(
            [
                np.sqrt(max(0.0, 1.0 - weight)) * empirical_vectors,
                np.sqrt(max(0.0, weight)) * semantic_vectors,
            ]
        )
    )
    neighbors = int(getattr(nn_config, "parsimony_cluster_neighbors", 20))
    empirical_pairs = _parsimony_mutual_neighbor_pairs(
        empirical_vectors,
        neighbors=neighbors,
    )
    semantic_pairs = _parsimony_mutual_neighbor_pairs(
        semantic_vectors,
        neighbors=neighbors,
    )
    candidate_pairs = sorted(empirical_pairs | semantic_pairs)
    missingness_weight = float(
        getattr(nn_config, "parsimony_cluster_missingness_weight", 0.15)
    )
    empirical_min = float(
        getattr(nn_config, "parsimony_cluster_empirical_min_similarity", 0.30)
    )
    strong_empirical = float(
        getattr(nn_config, "parsimony_cluster_strong_empirical_threshold", 0.80)
    )
    combined_threshold = float(
        getattr(nn_config, "parsimony_cluster_combined_threshold", 0.60)
    )
    pair_details: Dict[Tuple[int, int], Dict[str, Any]] = {}
    edges: List[Tuple[int, int]] = []
    for left_idx, right_idx in candidate_pairs:
        detail = _parsimony_pair_association(
            train_df=train_df,
            left=specs[left_idx],
            right=specs[right_idx],
            missingness_weight=missingness_weight,
        )
        semantic = float(
            np.clip(np.dot(semantic_vectors[left_idx], semantic_vectors[right_idx]), 0.0, 1.0)
        )
        combined = (1.0 - weight) * detail["empirical_similarity"] + weight * semantic
        detail.update(
            {
                "semantic_similarity": semantic,
                "combined_similarity": float(combined),
                "candidate_sources": sorted(
                    [
                        source
                        for source, pairs in [
                            ("empirical_value_knn", empirical_pairs),
                            ("semantic_knn", semantic_pairs),
                        ]
                        if (left_idx, right_idx) in pairs
                    ]
                ),
            }
        )
        qualifies = detail["empirical_similarity"] >= strong_empirical or (
            detail["empirical_similarity"] >= empirical_min
            and combined >= combined_threshold
        )
        detail["edge_retained"] = bool(qualifies)
        pair_details[(left_idx, right_idx)] = detail
        if qualifies:
            edges.append((left_idx, right_idx))

    parent = list(range(n_specs))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left_idx, right_idx in edges:
        union(left_idx, right_idx)
    components: Dict[int, List[int]] = {}
    connected_nodes = {index for edge in edges for index in edge}
    for index in sorted(connected_nodes):
        components.setdefault(find(index), []).append(index)

    min_size = int(getattr(nn_config, "parsimony_cluster_min_size", 2))
    max_size = int(getattr(nn_config, "parsimony_cluster_max_size", 12))
    split_components: List[List[int]] = []
    for component in components.values():
        if len(component) < min_size:
            continue
        split_components.extend(
            _parsimony_split_large_component(
                component,
                vectors=combined_vectors,
                max_size=max_size,
                random_state=random_state + len(split_components),
            )
        )

    clusters: List[Dict[str, Any]] = []
    for component in split_components:
        if len(component) < min_size:
            continue
        component = sorted(component, key=lambda idx: specs[idx].name)
        exact_pairs: List[Dict[str, Any]] = []
        for left_idx, right_idx in combinations(component, 2):
            key = tuple(sorted((left_idx, right_idx)))
            detail = pair_details.get(key)
            if detail is None:
                detail = _parsimony_pair_association(
                    train_df=train_df,
                    left=specs[left_idx],
                    right=specs[right_idx],
                    missingness_weight=missingness_weight,
                )
                semantic = float(
                    np.clip(
                        np.dot(semantic_vectors[left_idx], semantic_vectors[right_idx]),
                        0.0,
                        1.0,
                    )
                )
                detail.update(
                    {
                        "semantic_similarity": semantic,
                        "combined_similarity": float(
                            (1.0 - weight) * detail["empirical_similarity"]
                            + weight * semantic
                        ),
                        "candidate_sources": [],
                        "edge_retained": False,
                    }
                )
            exact_pairs.append(detail)
        empirical_cohesion = float(
            np.mean([item["empirical_similarity"] for item in exact_pairs])
        )
        combined_cohesion = float(
            np.mean([item["combined_similarity"] for item in exact_pairs])
        )
        if empirical_cohesion < empirical_min:
            continue
        member_names = [specs[index].name for index in component]
        clusters.append(
            {
                "member_names": member_names,
                "n_members": int(len(member_names)),
                "empirical_cohesion": empirical_cohesion,
                "combined_cohesion": combined_cohesion,
                "pair_associations": exact_pairs,
                "value_summaries": [value_summaries[index] for index in component],
            }
        )
    clusters.sort(key=lambda item: tuple(item["member_names"]))
    for index, cluster in enumerate(clusters, start=1):
        cluster["cluster_id"] = f"value_cluster_{index:03d}"
    return {
        "generation": {
            "uses_actual_extracted_values": True,
            "uses_treatment_or_outcome_labels": False,
            "n_rows": int(len(train_df)),
            "n_features": int(n_specs),
            "empirical_sketch_dim": int(
                getattr(nn_config, "parsimony_cluster_sketch_dim", 32)
            ),
            "neighbors_per_view": neighbors,
            "n_empirical_neighbor_pairs": int(len(empirical_pairs)),
            "n_semantic_neighbor_pairs": int(len(semantic_pairs)),
            "n_candidate_pairs": int(len(candidate_pairs)),
            "n_retained_graph_edges": int(len(edges)),
            "n_clusters": int(len(clusters)),
            "semantic_weight": weight,
            "empirical_min_similarity": empirical_min,
            "strong_empirical_threshold": strong_empirical,
            "combined_threshold": combined_threshold,
            "missingness_weight": missingness_weight,
            "continuous_association": "absolute_spearman",
            "categorical_association": "bias_corrected_cramers_v",
            "mixed_association": "correlation_ratio",
        },
        "clusters": clusters,
    }


def _explicit_feature_series(df: pd.DataFrame, spec: ExplicitFeatureSpec) -> pd.Series:
    value_col = f"explicit_feat_{spec.name}"
    legacy_col = f"explicit_conf_{spec.name}"
    if value_col in df.columns:
        return df[value_col]
    if legacy_col in df.columns:
        return df[legacy_col]
    return pd.Series([np.nan] * len(df), index=df.index, dtype="object")


def _explicit_feature_missing_mask(
    df: pd.DataFrame,
    spec: ExplicitFeatureSpec,
    values: Optional[pd.Series] = None,
) -> np.ndarray:
    value_col = f"explicit_feat_{spec.name}"
    legacy_col = f"explicit_conf_{spec.name}"
    source_col = value_col if value_col in df.columns else legacy_col
    missing_col = f"{source_col}_missing"
    if missing_col in df.columns:
        return df[missing_col].astype(bool).to_numpy()
    if values is None:
        values = _explicit_feature_series(df, spec)
    return values.isna().to_numpy()


def _parsimony_role_guard(
    current_specs: Sequence[ExplicitFeatureSpec],
    trial_specs: Sequence[ExplicitFeatureSpec],
) -> Optional[str]:
    if not trial_specs:
        return "would_remove_all_features"
    for role in ["confounder", "effect_modifier"]:
        had_role = any(role in spec.roles for spec in current_specs)
        keeps_role = any(role in spec.roles for spec in trial_specs)
        if had_role and not keeps_role:
            return f"would_remove_all_{role}_features"
    return None


def _parsimony_metric_snapshot(diagnostic: Dict[str, Any]) -> Dict[str, Any]:
    metrics = diagnostic.get("metrics", {}) if isinstance(diagnostic, dict) else {}
    keys = [
        "n_selected_features",
        "n_w_features",
        "n_x_features",
        "treatment_auroc",
        "treatment_brier",
        "treatment_log_loss",
        "outcome_auroc",
        "outcome_brier",
        "outcome_log_loss",
        "outcome_rmse",
        "r_loss_mean",
        "r_loss_relative_improvement",
        "tau_hat_pseudo_target_corr",
    ]
    return {key: metrics.get(key) for key in keys if key in metrics}


def _dedupe_strings(values: Sequence[Any]) -> List[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value)))


def _parsimony_context_fingerprint(context: Dict[str, Any]) -> str:
    payload = json.dumps(
        context,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def _parsimony_operational_factor_description(factor: Dict[str, Any]) -> str:
    supporting = "; ".join(str(value) for value in factor.get("supporting_indicators", []))
    contrary = "; ".join(str(value) for value in factor.get("contrary_indicators", []))
    return (
        f"{str(factor.get('description') or '').strip()} "
        f"Inference kind: {str(factor.get('inference_kind') or '').strip()}. "
        f"Measurement unit: {str(factor.get('unit') or 'not applicable').strip()}. "
        "Use only evidence temporally available before the treatment decision. "
        f"Supporting indicators: {supporting or 'none supplied'}. "
        f"Contrary indicators: {contrary or 'none supplied'}. "
        f"Minimum evidence for a non-null value: "
        f"{str(factor.get('minimum_evidence') or '').strip()}. "
        f"Null policy: {str(factor.get('null_policy') or '').strip()}"
    ).strip()


def _validate_parsimony_factor_candidate(
    *,
    response: Any,
    context: Dict[str, Any],
    cluster: Dict[str, Any],
    current_specs: Sequence[ExplicitFeatureSpec],
    required_names: set,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    reasons: List[str] = []
    if not isinstance(response, dict):
        return None, {"passed": False, "reasons": ["response_not_object"]}
    decision = str(response.get("decision") or "").strip().lower()
    if decision == "retain_cluster":
        return None, {
            "passed": True,
            "decision": "retain_cluster",
            "reasons": [str(response.get("rationale") or "agent_retained_cluster")],
        }
    if decision != "replace_cluster":
        return None, {"passed": False, "reasons": ["invalid_decision"]}
    if str(response.get("cluster_id") or "") != str(context.get("cluster_id") or ""):
        reasons.append("cluster_id_mismatch")

    member_names = set(cluster.get("member_names", []))
    replaceable = member_names - set(required_names)
    replaces = _dedupe_strings(
        [_normalize_feature_name(value) for value in response.get("replaces", []) or []]
    )
    if len(replaces) < 2:
        reasons.append("must_replace_at_least_two_members")
    if any(name not in replaceable for name in replaces):
        reasons.append("replacement_contains_unknown_or_protected_member")

    factors = response.get("factors", [])
    if not isinstance(factors, list):
        factors = []
        reasons.append("factors_not_list")
    max_factors = int(context.get("max_factors", 2) or 2)
    if not 1 <= len(factors) <= max_factors:
        reasons.append("factor_count_out_of_range")
    if factors and len(factors) >= len(replaces):
        reasons.append("factor_count_does_not_reduce_spec_count")

    current_by_name = {spec.name: spec for spec in current_specs}
    seen_names: set = set()
    factor_specs: List[ExplicitFeatureSpec] = []
    for factor in factors:
        if not isinstance(factor, dict):
            reasons.append("factor_not_object")
            continue
        name = _normalize_feature_name(factor.get("name", ""))
        if not name or name in seen_names:
            reasons.append("missing_or_duplicate_factor_name")
            continue
        seen_names.add(name)
        if name in current_by_name and name not in replaces:
            reasons.append(f"factor_name_conflicts_with_existing_feature:{name}")
            continue
        inference_kind = str(factor.get("inference_kind") or "").strip().lower()
        feature_type = str(factor.get("type") or "").strip().lower()
        categories = factor.get("categories")
        roles = _dedupe_strings(factor.get("roles", []) or [])
        if inference_kind not in {"implicit", "direct"}:
            reasons.append(f"invalid_inference_kind:{name}")
            continue
        if feature_type not in {"categorical", "continuous"}:
            reasons.append(f"invalid_factor_type:{name}")
            continue
        if inference_kind == "implicit" and feature_type != "categorical":
            reasons.append(f"implicit_factor_must_be_categorical:{name}")
            continue
        if feature_type == "categorical":
            if not isinstance(categories, list) or not 2 <= len(categories) <= 8:
                reasons.append(f"invalid_factor_categories:{name}")
                continue
            categories = _dedupe_strings(categories)
            if not 2 <= len(categories) <= 8:
                reasons.append(f"invalid_factor_categories:{name}")
                continue
        else:
            categories = None
            if not str(factor.get("unit") or "").strip():
                reasons.append(f"continuous_factor_missing_unit:{name}")
                continue
        if not roles or set(roles) - {"confounder", "effect_modifier"}:
            reasons.append(f"invalid_factor_roles:{name}")
            continue
        required_fields = [
            "description",
            "minimum_evidence",
            "null_policy",
            "rationale",
        ]
        if any(not str(factor.get(field) or "").strip() for field in required_fields):
            reasons.append(f"incomplete_operational_rubric:{name}")
            continue
        if not isinstance(factor.get("supporting_indicators"), list) or not factor.get(
            "supporting_indicators"
        ):
            reasons.append(f"missing_supporting_indicators:{name}")
            continue
        if not isinstance(factor.get("contrary_indicators"), list):
            reasons.append(f"invalid_contrary_indicators:{name}")
            continue
        try:
            factor_specs.append(
                ExplicitFeatureSpec(
                    name=name,
                    type=feature_type,
                    categories=categories,
                    roles=roles,
                    description=_parsimony_operational_factor_description(factor),
                )
            )
        except ValueError as exc:
            reasons.append(f"invalid_factor_spec:{name}:{exc}")

    expected_roles = {
        role
        for name in replaces
        if name in current_by_name
        for role in current_by_name[name].roles
    }
    actual_roles = {role for spec in factor_specs for role in spec.roles}
    if expected_roles != actual_roles:
        reasons.append(
            "factor_role_union_mismatch:"
            f"expected={sorted(expected_roles)}:actual={sorted(actual_roles)}"
        )
    passed = not reasons and bool(factor_specs)
    validation = {
        "passed": bool(passed),
        "decision": "replace_cluster",
        "reasons": reasons or ["valid_operational_factor_replacement"],
        "replaces": replaces,
        "factor_names": [spec.name for spec in factor_specs],
        "required_role_union": sorted(expected_roles),
    }
    if not passed:
        return None, validation
    return {
        "cluster_id": str(cluster.get("cluster_id")),
        "cluster": cluster,
        "replaces": replaces,
        "factor_specs": factor_specs,
        "raw_response": response,
    }, validation


def _parsimony_factor_extraction_quality(
    *,
    train_df: pd.DataFrame,
    factor_specs: Sequence[ExplicitFeatureSpec],
    min_coverage: float,
) -> Dict[str, Any]:
    summaries: List[Dict[str, Any]] = []
    reasons: List[str] = []
    for spec in factor_specs:
        values = _explicit_feature_series(train_df, spec)
        missing = _explicit_feature_missing_mask(train_df, spec, values)
        observed = values.loc[~missing]
        coverage = float(1.0 - np.mean(missing)) if len(train_df) else 0.0
        unique = int(observed.nunique(dropna=True))
        item = {
            "name": spec.name,
            "coverage": coverage,
            "n_unique_observed": unique,
            "required_min_coverage": float(min_coverage),
        }
        summaries.append(item)
        if coverage < float(min_coverage):
            reasons.append(f"factor_coverage_below_minimum:{spec.name}")
        if unique < 2:
            reasons.append(f"factor_has_insufficient_variation:{spec.name}")
        if spec.type == "continuous":
            numeric = pd.to_numeric(observed, errors="coerce").dropna()
            if len(numeric) < 2 or float(numeric.std(ddof=0)) <= 0.0:
                reasons.append(f"continuous_factor_is_constant:{spec.name}")
    return {
        "passed": not reasons,
        "reasons": _dedupe_strings(reasons),
        "factors": summaries,
    }


def _apply_parsimony_factor_replacements(
    specs: Sequence[ExplicitFeatureSpec],
    candidates: Sequence[Dict[str, Any]],
) -> List[ExplicitFeatureSpec]:
    removed = {
        name
        for candidate in candidates
        for name in candidate.get("replaces", [])
    }
    factors = [
        spec
        for candidate in candidates
        for spec in candidate.get("factor_specs", [])
    ]
    return _dedupe_specs([*[spec for spec in specs if spec.name not in removed], *factors])


def _parsimony_model_dimension(diagnostic: Dict[str, Any]) -> int:
    metrics = diagnostic.get("metrics", {}) if isinstance(diagnostic, dict) else {}
    return int(metrics.get("n_w_features", 0) or 0) + int(metrics.get("n_x_features", 0) or 0)


def _strict_parsimony_replacement_decision(
    *,
    base_diagnostic: Dict[str, Any],
    trial_diagnostic: Dict[str, Any],
    base_gate: Dict[str, Any],
    trial_gate: Dict[str, Any],
    epsilon: float,
) -> Tuple[bool, List[str], Dict[str, Any]]:
    base_metrics = base_diagnostic.get("metrics", {})
    trial_metrics = trial_diagnostic.get("metrics", {})
    reasons: List[str] = []
    deltas: Dict[str, Any] = {}
    if trial_metrics.get("status") != "ok":
        reasons.append("trial_diagnostic_status_not_ok")
    if int(trial_gate.get("n_failed_criteria", 0) or 0) > int(
        base_gate.get("n_failed_criteria", 0) or 0
    ):
        reasons.append("review_gate_would_worsen")

    higher_is_better = ["treatment_auroc", "outcome_auroc"]
    lower_is_better = [
        "treatment_brier",
        "treatment_log_loss",
        "outcome_brier",
        "outcome_log_loss",
        "outcome_rmse",
        "r_loss_mean",
    ]
    for metric in higher_is_better:
        base_value = _finite_or_none(base_metrics.get(metric))
        if base_value is None:
            continue
        trial_value = _finite_or_none(trial_metrics.get(metric))
        if trial_value is None:
            reasons.append(f"{metric}_missing_after_replacement")
            continue
        delta = float(trial_value - base_value)
        deltas[metric] = delta
        if delta < -float(epsilon):
            reasons.append(f"{metric}_degraded")
    for metric in lower_is_better:
        base_value = _finite_or_none(base_metrics.get(metric))
        if base_value is None:
            continue
        trial_value = _finite_or_none(trial_metrics.get(metric))
        if trial_value is None:
            reasons.append(f"{metric}_missing_after_replacement")
            continue
        delta = float(trial_value - base_value)
        deltas[metric] = delta
        if delta > float(epsilon):
            reasons.append(f"{metric}_degraded")
    if not reasons:
        reasons.append("strict_pareto_non_degradation")
    return reasons == ["strict_pareto_non_degradation"], reasons, deltas


def _parsimony_replacement_evaluation_row(
    *,
    outer_fold: int,
    phase: str,
    result: Dict[str, Any],
) -> Dict[str, Any]:
    cluster_ids = result.get("cluster_ids")
    if not cluster_ids and result.get("cluster_id"):
        cluster_ids = [result.get("cluster_id")]
    return {
        "schema_version": _PARSIMONY_SCHEMA_VERSION,
        "outer_fold": int(outer_fold),
        "event": "parsimony_replacement_evaluation",
        "phase": str(phase),
        "cluster_ids": list(cluster_ids or []),
        "allowed": bool(result.get("allowed", False)),
        "reasons": list(result.get("reasons", [])),
        "metric_deltas": result.get("metric_deltas", {}),
        "base_dimension": result.get("base_dimension"),
        "trial_dimension": result.get("trial_dimension"),
        "dimension_reduction": result.get("dimension_reduction"),
        "metrics_after": _parsimony_metric_snapshot(result.get("diagnostic", {})),
        "gate_after": result.get("gate", {}),
        "selected_features_after": [
            _spec_to_dict(spec) for spec in result.get("trial_specs", [])
        ],
    }


def _parsimony_factor_agent_worker(
    search_config: AgenticFeatureSearchConfig,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    agent = make_feature_search_agent(search_config)
    try:
        response = agent.propose(context)
        result: Dict[str, Any] = {"response": response}
        trace = _get_agent_response_trace(agent)
        if trace is not None:
            result["agent_raw_output"] = trace
        return result
    except Exception as exc:
        return {
            "response": None,
            "error": f"{exc.__class__.__name__}: {exc}",
        }


def _cluster_relevant_htr_evidence(
    evidence: Dict[str, Any],
    specs: Sequence[ExplicitFeatureSpec],
) -> Dict[str, Any]:
    compact = _compact_htr_attention_evidence(evidence)
    token_source = " ".join(_parsimony_feature_contract_document(spec) for spec in specs)
    tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", token_source.lower())
        if len(token) >= 4 and token not in ENGLISH_STOP_WORDS
    }
    result: Dict[str, Any] = {}
    for stage in ["nuisance", "effect", "pair_uplift"]:
        payload = compact.get(stage)
        if not isinstance(payload, dict):
            continue
        rows = list(payload.get("attention", []) or [])
        scored = []
        for index, row in enumerate(rows):
            text = json.dumps(row, default=_json_default).lower()
            overlap = sum(1 for token in tokens if token in text)
            scored.append((overlap, -index, row))
        selected = [row for score, _, row in sorted(scored, reverse=True) if score > 0][:8]
        if not selected:
            selected = rows[:3]
        result[stage] = {
            "metrics": payload.get("metrics", {}),
            "attention": selected,
        }
    if result:
        result["selection_policy"] = "member-token overlap, up to 8 rows per stage"
    return result


def _parsimony_removal_decision(
    *,
    base_diagnostic: Dict[str, Any],
    trial_diagnostic: Dict[str, Any],
    base_gate: Dict[str, Any],
    trial_gate: Dict[str, Any],
    nn_config: MultiModelAgenticForestConfig,
) -> Tuple[bool, List[str], Dict[str, Any]]:
    base_metrics = base_diagnostic.get("metrics", {})
    trial_metrics = trial_diagnostic.get("metrics", {})
    auc_tolerance = float(getattr(nn_config, "parsimony_review_auc_tolerance", 0.01))
    loss_tolerance = float(getattr(nn_config, "parsimony_review_loss_relative_tolerance", 0.03))
    reasons: List[str] = []
    base_failures = int(base_gate.get("n_failed_criteria", 0) or 0)
    trial_failures = int(trial_gate.get("n_failed_criteria", 0) or 0)
    if trial_failures > base_failures:
        reasons.append("review_gate_would_worsen")
    deltas: Dict[str, Any] = {}
    for metric in ["treatment_auroc", "outcome_auroc"]:
        base_value = _finite_or_none(base_metrics.get(metric))
        trial_value = _finite_or_none(trial_metrics.get(metric))
        if base_value is None or trial_value is None:
            continue
        delta = float(trial_value - base_value)
        deltas[metric] = delta
        if delta < -auc_tolerance:
            reasons.append(f"{metric}_drop_exceeds_tolerance")
    for metric in ["treatment_log_loss", "outcome_log_loss", "outcome_rmse", "r_loss_mean"]:
        base_value = _finite_or_none(base_metrics.get(metric))
        trial_value = _finite_or_none(trial_metrics.get(metric))
        if base_value is None or trial_value is None or base_value <= 0.0:
            continue
        relative_change = float((trial_value - base_value) / base_value)
        deltas[f"{metric}_relative_change"] = relative_change
        if relative_change > loss_tolerance:
            reasons.append(f"{metric}_increase_exceeds_tolerance")
    if not reasons:
        reasons.append("within_parsimony_tolerances")
    return reasons == ["within_parsimony_tolerances"], reasons, deltas


def _extracted_feature_review_benchmarks(
    bow_metrics: Dict[str, Any],
    embedding_evidence: Dict[str, Any],
) -> Dict[str, Any]:
    treatment_auc_values = _collect_metric_values(bow_metrics, "treatment_auroc")
    outcome_auc_values = _collect_metric_values(bow_metrics, "outcome_auroc")
    treatment_log_losses = _collect_metric_values(bow_metrics, "treatment_log_loss")
    outcome_log_losses = _collect_metric_values(bow_metrics, "outcome_log_loss")
    outcome_rmses = _collect_metric_values(bow_metrics, "outcome_rmse")
    r_losses = _collect_metric_values(bow_metrics, "r_loss_mean")

    embedding_probe_auc = _embedding_probe_auc_benchmarks(embedding_evidence)
    if embedding_probe_auc.get("treatment_probe_auc") is not None:
        treatment_auc_values.append(float(embedding_probe_auc["treatment_probe_auc"]))
    if embedding_probe_auc.get("outcome_probe_auc") is not None:
        outcome_auc_values.append(float(embedding_probe_auc["outcome_probe_auc"]))

    return {
        "treatment_auroc": _max_or_none(treatment_auc_values),
        "outcome_auroc": _max_or_none(outcome_auc_values),
        "treatment_log_loss": _min_or_none(treatment_log_losses),
        "outcome_log_loss": _min_or_none(outcome_log_losses),
        "outcome_rmse": _min_or_none(outcome_rmses),
        "r_loss_mean": _min_or_none(r_losses),
        "embedding_probe_auc": embedding_probe_auc,
    }


def _collect_metric_values(payload: Any, metric_name: str) -> List[float]:
    values: List[float] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == metric_name or str(key).endswith(f"_{metric_name}"):
                numeric = _finite_or_none(value)
                if numeric is not None:
                    values.append(float(numeric))
            values.extend(_collect_metric_values(value, metric_name))
    elif isinstance(payload, list):
        for item in payload:
            values.extend(_collect_metric_values(item, metric_name))
    return values


def _embedding_probe_auc_benchmarks(evidence: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(evidence, dict):
        return {}
    treatment_values: List[float] = []
    outcome_values: List[float] = []
    effect_values: List[float] = []
    for contrast in evidence.get("contrasts", []) or []:
        if not isinstance(contrast, dict):
            continue
        auc = _finite_or_none(contrast.get("probe_auc"))
        if auc is None:
            continue
        name = str(contrast.get("name", ""))
        family = str(contrast.get("contrast_family", ""))
        role_hint = str(contrast.get("role_hint", ""))
        if name == "treatment":
            treatment_values.append(float(auc))
        elif name == "outcome":
            outcome_values.append(float(auc))
        elif role_hint == "effect_modifier" or "r_pseudo" in family:
            effect_values.append(float(auc))
    return {
        "treatment_probe_auc": _max_or_none(treatment_values),
        "outcome_probe_auc": _max_or_none(outcome_values),
        "effect_modifier_probe_auc": _max_or_none(effect_values),
    }


def _extracted_feature_review_gate(
    *,
    diagnostic: Dict[str, Any],
    nn_config: MultiModelAgenticForestConfig,
) -> Dict[str, Any]:
    metrics = diagnostic.get("metrics", {})
    benchmark = diagnostic.get("benchmark", {})
    failures: List[Dict[str, Any]] = []
    if metrics.get("status") != "ok":
        failures.append(
            {
                "metric": "status",
                "observed": metrics.get("status"),
                "benchmark": "ok",
                "reason": "diagnostic_status_not_ok",
            }
        )

    auc_margin = float(getattr(nn_config, "extracted_feature_review_auc_margin", 0.02))
    loss_margin = float(getattr(nn_config, "extracted_feature_review_loss_relative_margin", 0.05))
    min_auc = float(getattr(nn_config, "extracted_feature_review_min_benchmark_auc", 0.55))

    for metric in ["treatment_auroc", "outcome_auroc"]:
        observed = _finite_or_none(metrics.get(metric))
        target = _finite_or_none(benchmark.get(metric))
        if target is None or target < min_auc:
            continue
        if observed is None or observed < target - auc_margin:
            failures.append(
                {
                    "metric": metric,
                    "observed": observed,
                    "benchmark": target,
                    "required_min": target - auc_margin,
                    "reason": "auc_under_benchmark",
                }
            )

    for metric in ["treatment_log_loss", "outcome_log_loss", "outcome_rmse", "r_loss_mean"]:
        observed = _finite_or_none(metrics.get(metric))
        target = _finite_or_none(benchmark.get(metric))
        if target is None or target <= 0.0:
            continue
        max_allowed = target * (1.0 + loss_margin)
        if observed is None or observed > max_allowed:
            failures.append(
                {
                    "metric": metric,
                    "observed": observed,
                    "benchmark": target,
                    "required_max": max_allowed,
                    "reason": "loss_over_benchmark",
                }
            )

    return {
        "passed": not failures,
        "failed_criteria": failures,
        "n_failed_criteria": int(len(failures)),
    }


def _extracted_review_selection_score(
    diagnostic: Dict[str, Any],
    gate: Dict[str, Any],
) -> Tuple[int, float, float, float]:
    metrics = diagnostic.get("metrics", {})
    fail_count = int(gate.get("n_failed_criteria", 0))
    r_loss = _finite_or_none(metrics.get("r_loss_mean"))
    treatment_auc = _finite_or_none(metrics.get("treatment_auroc"))
    outcome_auc = _finite_or_none(metrics.get("outcome_auroc"))
    return (
        fail_count,
        float("inf") if r_loss is None else float(r_loss),
        float("inf") if treatment_auc is None else -float(treatment_auc),
        float("inf") if outcome_auc is None else -float(outcome_auc),
    )


def _extracted_review_summary(
    *,
    diagnostic: Optional[Dict[str, Any]],
    status: str,
    passed: bool,
    rounds: int,
) -> Dict[str, Any]:
    metrics = diagnostic.get("metrics", {}) if diagnostic else {}
    gate = diagnostic.get("gate", {}) if diagnostic else {}
    return {
        "enabled": True,
        "review_status": status,
        "review_passed": bool(passed),
        "review_rounds": int(rounds),
        "n_failed_criteria": int(gate.get("n_failed_criteria", 0) or 0),
        "failed_criteria": gate.get("failed_criteria", []),
        "treatment_auroc": metrics.get("treatment_auroc"),
        "outcome_auroc": metrics.get("outcome_auroc"),
        "outcome_rmse": metrics.get("outcome_rmse"),
        "r_loss_mean": metrics.get("r_loss_mean"),
        "r_loss_relative_improvement": metrics.get("r_loss_relative_improvement"),
    }


def _protect_required_feature_proposals(
    proposals: Sequence[AgenticFeatureProposal],
    required_names: set,
) -> Tuple[List[AgenticFeatureProposal], List[Dict[str, Any]]]:
    kept: List[AgenticFeatureProposal] = []
    rejected: List[Dict[str, Any]] = []
    for proposal in proposals:
        if proposal.action == "remove" and proposal.name in required_names:
            rejected.append(
                {
                    "proposal": _proposal_to_dict(proposal),
                    "reason": "cannot_remove_required_feature",
                }
            )
            continue
        kept.append(proposal)
    return kept, rejected


def _spec_sets_differ(
    left: Sequence[ExplicitFeatureSpec],
    right: Sequence[ExplicitFeatureSpec],
) -> bool:
    return [_spec_to_dict(spec) for spec in left] != [_spec_to_dict(spec) for spec in right]


def _redact_review_artifact(
    diagnostic: Dict[str, Any],
    search_config: AgenticFeatureSearchConfig,
) -> Dict[str, Any]:
    del search_config
    return diagnostic


def _compact_extracted_feature_review_context(context: Dict[str, Any]) -> Dict[str, Any]:
    compact = dict(context)
    original = compact.get("original_bow_context")
    if isinstance(original, dict) and isinstance(original.get("feature_importance"), dict):
        compact["original_bow_context"] = {
            **original,
            "feature_importance": _compact_multi_model_importance(original["feature_importance"]),
        }
    if isinstance(compact.get("embedding_contrast_evidence"), dict):
        compact["embedding_contrast_evidence"] = _compact_embedding_contrast_evidence(
            compact["embedding_contrast_evidence"]
        )
    if isinstance(compact.get("htr_attention_evidence"), dict):
        compact["htr_attention_evidence"] = _compact_htr_attention_evidence(
            compact["htr_attention_evidence"]
        )
    return _round_floats(compact)


def _max_or_none(values: Sequence[float]) -> Optional[float]:
    finite = [float(value) for value in values if np.isfinite(value)]
    return max(finite) if finite else None


def _min_or_none(values: Sequence[float]) -> Optional[float]:
    finite = [float(value) for value in values if np.isfinite(value)]
    return min(finite) if finite else None


def _fit_transform_bow_plus_explicit(
    *,
    texts: Sequence[str],
    fit_pos: np.ndarray,
    heldout_pos: np.ndarray,
    vectorizer_params: Dict[str, Any],
    explicit_feature_dicts: Optional[List[Dict[str, Any]]],
    explicit_specs: Optional[List[ExplicitFeatureSpec]],
):
    vectorizer = _make_bow_vectorizer(vectorizer_params)
    x_fit = vectorizer.fit_transform([texts[i] for i in fit_pos])
    x_heldout = vectorizer.transform([texts[i] for i in heldout_pos])
    if not explicit_feature_dicts or not explicit_specs:
        return x_fit, x_heldout

    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}
    fit_dicts = [explicit_feature_dicts[int(i)] for i in fit_pos]
    heldout_dicts = [explicit_feature_dicts[int(i)] for i in heldout_pos]
    fit_explicit, _ = get_raw_explicit_features(
        fit_dicts,
        explicit_specs,
        continuous_means=means,
        continuous_stds=stds,
        role=None,
    )
    heldout_explicit, _ = get_raw_explicit_features(
        heldout_dicts,
        explicit_specs,
        continuous_means=means,
        continuous_stds=stds,
        role=None,
    )
    return (
        _hstack_sparse_and_dense(x_fit, fit_explicit),
        _hstack_sparse_and_dense(x_heldout, heldout_explicit),
    )


def _append_explicit_features_full(
    x_text,
    text_feature_names: np.ndarray,
    *,
    explicit_feature_dicts: Optional[List[Dict[str, Any]]],
    explicit_specs: Optional[List[ExplicitFeatureSpec]],
) -> Tuple[Any, np.ndarray, List[str]]:
    if not explicit_feature_dicts or not explicit_specs:
        return x_text, text_feature_names, []
    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}
    explicit_features, explicit_names = get_raw_explicit_features(
        explicit_feature_dicts,
        explicit_specs,
        continuous_means=means,
        continuous_stds=stds,
        role=None,
    )
    prefixed_names = [f"explicit:{name}" for name in explicit_names]
    features = np.concatenate([text_feature_names, np.asarray(prefixed_names, dtype=object)])
    return _hstack_sparse_and_dense(x_text, explicit_features), features, prefixed_names


def _hstack_sparse_and_dense(x_text: Any, explicit_features: Sequence[Sequence[float]]):
    explicit_matrix = np.asarray(explicit_features, dtype=np.float32)
    if explicit_matrix.ndim != 2 or explicit_matrix.shape[1] == 0:
        return x_text
    return sparse.hstack(
        [x_text, sparse.csr_matrix(explicit_matrix)],
        format="csr",
        dtype=np.float32,
    )


def _fit_binary_bow_fold(
    texts: Sequence[str],
    labels: np.ndarray,
    fit_pos: np.ndarray,
    heldout_pos: np.ndarray,
    vectorizer_params: Dict[str, Any],
    model_params: Dict[str, Any],
    *,
    explicit_feature_dicts: Optional[List[Dict[str, Any]]] = None,
    explicit_specs: Optional[List[ExplicitFeatureSpec]] = None,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels).astype(int)
    fit_pos = np.asarray(fit_pos)
    heldout_pos = np.asarray(heldout_pos)
    if len(np.unique(labels[fit_pos])) < 2:
        return heldout_pos, np.full(
            len(heldout_pos),
            float(np.mean(labels[fit_pos])),
            dtype=float,
        )
    x_fit, x_heldout = _fit_transform_bow_plus_explicit(
        texts=texts,
        fit_pos=fit_pos,
        heldout_pos=heldout_pos,
        vectorizer_params=vectorizer_params,
        explicit_feature_dicts=explicit_feature_dicts,
        explicit_specs=explicit_specs,
    )
    model = _make_bow_classifier(model_params, random_state=random_state)
    model.fit(x_fit, labels[fit_pos])
    return heldout_pos, model.predict_proba(x_heldout)[:, 1]


def _fit_regression_bow_fold(
    texts: Sequence[str],
    values: np.ndarray,
    fit_pos: np.ndarray,
    heldout_pos: np.ndarray,
    vectorizer_params: Dict[str, Any],
    model_params: Dict[str, Any],
    *,
    explicit_feature_dicts: Optional[List[Dict[str, Any]]] = None,
    explicit_specs: Optional[List[ExplicitFeatureSpec]] = None,
    sample_weight: Optional[np.ndarray] = None,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    fit_pos = np.asarray(fit_pos)
    heldout_pos = np.asarray(heldout_pos)
    x_fit, x_heldout = _fit_transform_bow_plus_explicit(
        texts=texts,
        fit_pos=fit_pos,
        heldout_pos=heldout_pos,
        vectorizer_params=vectorizer_params,
        explicit_feature_dicts=explicit_feature_dicts,
        explicit_specs=explicit_specs,
    )
    model = _make_bow_regressor(model_params, random_state=random_state)
    fold_weight = None
    if sample_weight is not None:
        weights = np.asarray(sample_weight, dtype=float)
        fold_weight = weights[fit_pos]
    _fit_regressor(
        model,
        x_fit,
        values[fit_pos],
        sample_weight=fold_weight,
        unsupported_sample_weight_policy=str(
            model_params["unsupported_sample_weight_policy"]
        ),
    )
    return heldout_pos, model.predict(x_heldout)


def _fit_regressor(
    model: Any,
    x: Any,
    y: np.ndarray,
    *,
    sample_weight: Optional[np.ndarray] = None,
    unsupported_sample_weight_policy: str,
) -> Any:
    if unsupported_sample_weight_policy not in {
        "fail_closed",
        "unweighted_legacy_compatibility",
    }:
        raise ValueError("unsupported sample-weight policy")
    if sample_weight is None:
        return model.fit(x, y)
    weights = np.asarray(sample_weight, dtype=float)
    if weights.shape[0] != len(y):
        raise ValueError("sample_weight must have one value per training row")
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    if float(np.sum(weights)) <= 0.0:
        if unsupported_sample_weight_policy == "fail_closed":
            raise ValueError("sample_weight has no positive finite mass")
        return model.fit(x, y)
    try:
        return model.fit(x, y, sample_weight=weights)
    except TypeError as exc:
        if unsupported_sample_weight_policy == "fail_closed":
            raise TypeError(
                f"BoW regressor {type(model).__name__} does not accept configured "
                "sample weights"
            ) from exc
        logger.warning(
            "Legacy BoW compatibility: %s does not accept sample_weight; "
            "fitting unweighted",
            type(model).__name__,
        )
        return model.fit(x, y)


_BOW_VECTORIZER_PARAMETER_KEYS = frozenset(
    field.name for field in fields(TfidfVectorizerScientificConfig)
)
_BOW_MODEL_PARAMETER_KEYS = frozenset(
    {
        "bow_model",
        "logistic_c",
        "logistic_max_iter",
        "ridge_alpha",
        "logistic_scientific",
        "ridge_scientific",
        "forest_scientific",
        "xgboost_scientific",
        "single_class_policy",
        "empty_vocabulary_policy",
        "unsupported_sample_weight_policy",
    }
)


def _require_exact_constructor_mapping(
    params: Dict[str, Any],
    *,
    expected: frozenset[str],
    label: str,
) -> Dict[str, Any]:
    if not isinstance(params, dict):
        raise TypeError(f"{label} parameters must be a plain mapping")
    missing = sorted(expected - set(params))
    extra = sorted(set(params) - expected)
    if missing or extra:
        raise ValueError(
            f"{label} parameter mapping is not closed; "
            f"missing={missing}, extra={extra}"
        )
    return params


def _bow_vectorizer_params(view: BoWViewConfig) -> Dict[str, Any]:
    if type(view) is not BoWViewConfig:
        raise TypeError("BoW vectorizer parameters require BoWViewConfig")
    return asdict(view.vectorizer_scientific)


def _bow_model_params(view: BoWViewConfig) -> Dict[str, Any]:
    if type(view) is not BoWViewConfig:
        raise TypeError("BoW model parameters require BoWViewConfig")
    return {
        "bow_model": str(view.bow_model),
        "logistic_c": float(view.logistic_c),
        "logistic_max_iter": int(view.logistic_max_iter),
        "ridge_alpha": float(view.ridge_alpha),
        "logistic_scientific": asdict(view.logistic_scientific),
        "ridge_scientific": asdict(view.ridge_scientific),
        "forest_scientific": asdict(view.forest_scientific),
        "xgboost_scientific": asdict(view.xgboost_scientific),
        "single_class_policy": str(view.single_class_policy),
        "empty_vocabulary_policy": str(view.empty_vocabulary_policy),
        "unsupported_sample_weight_policy": str(
            view.unsupported_sample_weight_policy
        ),
    }


def _make_bow_vectorizer(params: Dict[str, Any]) -> TfidfVectorizer:
    params = _require_exact_constructor_mapping(
        params,
        expected=_BOW_VECTORIZER_PARAMETER_KEYS,
        label="BoW TF-IDF vectorizer",
    )
    if params["input_text_case_policy"] != "vectorizer_controls_complete_text_case_v1":
        raise ValueError("BoW vectorizer input text policy is unsupported")
    if params["preprocessor_policy"] != "none":
        raise ValueError("BoW vectorizer preprocessor policy is unsupported")
    if params["tokenizer_policy"] != "token_pattern":
        raise ValueError("BoW vectorizer tokenizer policy is unsupported")
    if params["vocabulary_policy"] != "fit_scope_only":
        raise ValueError("BoW vectorizer vocabulary policy is unsupported")
    if params["feature_selection_rule"] != "sklearn_term_frequency_rank_v1":
        raise ValueError("BoW vectorizer feature-selection rule is unsupported")
    dtype = {
        "float32": np.float32,
        "float64": np.float64,
    }[str(params["dtype"])]
    return TfidfVectorizer(
        input=str(params["input"]),
        encoding=str(params["encoding"]),
        decode_error=str(params["decode_error"]),
        strip_accents=params["strip_accents"],
        lowercase=bool(params["lowercase"]),
        preprocessor=None,
        tokenizer=None,
        analyzer=str(params["analyzer"]),
        stop_words=params["stop_words"],
        token_pattern=params["token_pattern"],
        ngram_range=(
            int(params["ngram_range_min"]),
            int(params["ngram_range_max"]),
        ),
        min_df=int(params["min_df"]),
        max_df=float(params["max_df"]),
        max_features=(
            None
            if params["max_features"] is None
            else int(params["max_features"])
        ),
        vocabulary=None,
        binary=bool(params["binary"]),
        dtype=dtype,
        norm=params["norm"],
        use_idf=bool(params["use_idf"]),
        smooth_idf=bool(params["smooth_idf"]),
        sublinear_tf=bool(params["sublinear_tf"]),
    )


def _make_bow_classifier(params: Dict[str, Any], *, random_state: int = 17):
    params = _require_exact_constructor_mapping(
        params,
        expected=_BOW_MODEL_PARAMETER_KEYS,
        label="BoW classifier",
    )
    model_name = str(params["bow_model"]).strip().lower()
    if model_name == "linear":
        config = dict(params["logistic_scientific"])
        return LogisticRegression(
            penalty=config["penalty"],
            dual=bool(config["dual"]),
            tol=float(config["tol"]),
            C=float(params["logistic_c"]),
            fit_intercept=bool(config["fit_intercept"]),
            intercept_scaling=float(config["intercept_scaling"]),
            class_weight=config["class_weight"],
            solver=str(config["solver"]),
            max_iter=int(params["logistic_max_iter"]),
            multi_class=str(config["multi_class"]),
            verbose=0,
            warm_start=bool(config["warm_start"]),
            n_jobs=1,
            l1_ratio=config["l1_ratio"],
            random_state=random_state,
        )
    config = dict(params["forest_scientific"])
    forest_kwargs = {
        "n_estimators": int(config["n_estimators"]),
        "criterion": str(config["classifier_criterion"]),
        "max_depth": config["max_depth"],
        "min_samples_split": config["min_samples_split"],
        "min_samples_leaf": config["min_samples_leaf"],
        "min_weight_fraction_leaf": float(config["min_weight_fraction_leaf"]),
        "max_features": config["max_features"],
        "max_leaf_nodes": config["max_leaf_nodes"],
        "min_impurity_decrease": float(config["min_impurity_decrease"]),
        "oob_score": bool(config["oob_score"]),
        "n_jobs": 1,
        "random_state": random_state,
        "verbose": 0,
        "warm_start": bool(config["warm_start"]),
        "class_weight": config["class_weight"],
        "ccp_alpha": float(config["ccp_alpha"]),
        "max_samples": config["max_samples"],
        "monotonic_cst": config["monotonic_cst"],
    }
    if model_name == "extratrees":
        return ExtraTreesClassifier(
            **forest_kwargs,
            bootstrap=bool(config["extra_trees_bootstrap"]),
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            **forest_kwargs,
            bootstrap=bool(config["random_forest_bootstrap"]),
        )
    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "bow_model='xgboost' requires xgboost; refusing learner substitution"
            ) from exc
        config = dict(params["xgboost_scientific"])
        return XGBClassifier(
            n_estimators=int(config["n_estimators"]),
            max_depth=int(config["max_depth"]),
            max_leaves=int(config["max_leaves"]),
            max_bin=int(config["max_bin"]),
            grow_policy=str(config["grow_policy"]),
            learning_rate=float(config["learning_rate"]),
            booster=str(config["booster"]),
            tree_method=str(config["tree_method"]),
            gamma=float(config["gamma"]),
            min_child_weight=float(config["min_child_weight"]),
            max_delta_step=float(config["max_delta_step"]),
            subsample=float(config["subsample"]),
            sampling_method=str(config["sampling_method"]),
            colsample_bytree=float(config["colsample_bytree"]),
            colsample_bylevel=float(config["colsample_bylevel"]),
            colsample_bynode=float(config["colsample_bynode"]),
            reg_alpha=float(config["reg_alpha"]),
            reg_lambda=float(config["reg_lambda"]),
            scale_pos_weight=float(config["scale_pos_weight"]),
            base_score=float(config["base_score"]),
            missing=np.nan,
            num_parallel_tree=int(config["num_parallel_tree"]),
            monotone_constraints=config["monotone_constraints"],
            interaction_constraints=config["interaction_constraints"],
            enable_categorical=bool(config["enable_categorical"]),
            max_cat_to_onehot=int(config["max_cat_to_onehot"]),
            max_cat_threshold=int(config["max_cat_threshold"]),
            multi_strategy=str(config["multi_strategy"]),
            objective=str(config["classifier_objective"]),
            eval_metric=str(config["classifier_eval_metric"]),
            random_state=random_state,
            n_jobs=1,
        )
    raise ValueError(f"Unsupported bow_model: {model_name}")


def _make_bow_regressor(params: Dict[str, Any], *, random_state: int = 17):
    params = _require_exact_constructor_mapping(
        params,
        expected=_BOW_MODEL_PARAMETER_KEYS,
        label="BoW regressor",
    )
    model_name = str(params["bow_model"]).strip().lower()
    if model_name == "linear":
        config = dict(params["ridge_scientific"])
        return Ridge(
            alpha=float(params["ridge_alpha"]),
            fit_intercept=bool(config["fit_intercept"]),
            copy_X=True,
            max_iter=config["max_iter"],
            tol=float(config["tol"]),
            solver=str(config["solver"]),
            positive=bool(config["positive"]),
            random_state=random_state,
        )
    config = dict(params["forest_scientific"])
    forest_kwargs = {
        "n_estimators": int(config["n_estimators"]),
        "criterion": str(config["regressor_criterion"]),
        "max_depth": config["max_depth"],
        "min_samples_split": config["min_samples_split"],
        "min_samples_leaf": config["min_samples_leaf"],
        "min_weight_fraction_leaf": float(config["min_weight_fraction_leaf"]),
        "max_features": config["max_features"],
        "max_leaf_nodes": config["max_leaf_nodes"],
        "min_impurity_decrease": float(config["min_impurity_decrease"]),
        "oob_score": bool(config["oob_score"]),
        "n_jobs": 1,
        "random_state": random_state,
        "verbose": 0,
        "warm_start": bool(config["warm_start"]),
        "ccp_alpha": float(config["ccp_alpha"]),
        "max_samples": config["max_samples"],
        "monotonic_cst": config["monotonic_cst"],
    }
    if model_name == "extratrees":
        return ExtraTreesRegressor(
            **forest_kwargs,
            bootstrap=bool(config["extra_trees_bootstrap"]),
        )
    if model_name == "random_forest":
        return RandomForestRegressor(
            **forest_kwargs,
            bootstrap=bool(config["random_forest_bootstrap"]),
        )
    if model_name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError(
                "bow_model='xgboost' requires xgboost; refusing learner substitution"
            ) from exc
        config = dict(params["xgboost_scientific"])
        return XGBRegressor(
            n_estimators=int(config["n_estimators"]),
            max_depth=int(config["max_depth"]),
            max_leaves=int(config["max_leaves"]),
            max_bin=int(config["max_bin"]),
            grow_policy=str(config["grow_policy"]),
            learning_rate=float(config["learning_rate"]),
            booster=str(config["booster"]),
            tree_method=str(config["tree_method"]),
            gamma=float(config["gamma"]),
            min_child_weight=float(config["min_child_weight"]),
            max_delta_step=float(config["max_delta_step"]),
            subsample=float(config["subsample"]),
            sampling_method=str(config["sampling_method"]),
            colsample_bytree=float(config["colsample_bytree"]),
            colsample_bylevel=float(config["colsample_bylevel"]),
            colsample_bynode=float(config["colsample_bynode"]),
            reg_alpha=float(config["reg_alpha"]),
            reg_lambda=float(config["reg_lambda"]),
            base_score=float(config["base_score"]),
            missing=np.nan,
            num_parallel_tree=int(config["num_parallel_tree"]),
            monotone_constraints=config["monotone_constraints"],
            interaction_constraints=config["interaction_constraints"],
            enable_categorical=bool(config["enable_categorical"]),
            max_cat_to_onehot=int(config["max_cat_to_onehot"]),
            max_cat_threshold=int(config["max_cat_threshold"]),
            multi_strategy=str(config["multi_strategy"]),
            objective=str(config["regressor_objective"]),
            eval_metric=str(config["regressor_eval_metric"]),
            random_state=random_state,
            n_jobs=1,
        )
    raise ValueError(f"Unsupported bow_model: {model_name}")


def _bounded_fold_count(requested: int, n_rows: int) -> int:
    if n_rows < 2:
        raise ValueError("At least two rows are required for cross-fitting")
    return max(2, min(int(requested), int(n_rows)))


def _bounded_stratified_folds(labels: np.ndarray, requested: int) -> int:
    values, counts = np.unique(labels.astype(int), return_counts=True)
    if len(values) < 2:
        raise ValueError("Binary cross-fitting requires both treatment/outcome classes")
    return max(2, min(int(requested), int(np.min(counts)), int(len(labels))))


def _binary_split_items(
    labels: np.ndarray,
    *,
    requested_folds: int,
    random_state: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    values, counts = np.unique(labels.astype(int), return_counts=True)
    if len(values) >= 2 and int(np.min(counts)) >= 2:
        folds = _bounded_stratified_folds(labels, requested_folds)
        splitter = StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=random_state,
        )
        return [
            (np.asarray(fit_pos), np.asarray(heldout_pos))
            for fit_pos, heldout_pos in splitter.split(np.zeros(len(labels)), labels)
        ]

    folds = _bounded_fold_count(requested_folds, len(labels))
    splitter = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    return [
        (np.asarray(fit_pos), np.asarray(heldout_pos))
        for fit_pos, heldout_pos in splitter.split(np.zeros(len(labels)))
    ]


def _top_feature_rows(
    features: np.ndarray,
    scores: np.ndarray,
    top_n: int,
    *,
    descending: bool = True,
    treatment_coef: Optional[np.ndarray] = None,
    outcome_coef: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    if len(features) == 0:
        return []
    order = np.argsort(scores)
    if descending:
        order = order[::-1]
    rows: List[Dict[str, Any]] = []
    for idx in order[:top_n]:
        row = {
            "feature": str(features[idx]),
            "score": _finite_or_none(scores[idx]),
        }
        if treatment_coef is not None:
            row["treatment_score"] = _finite_or_none(treatment_coef[idx])
            row["abs_treatment_score"] = _finite_or_none(abs(treatment_coef[idx]))
        if outcome_coef is not None:
            row["outcome_score"] = _finite_or_none(outcome_coef[idx])
            row["abs_outcome_score"] = _finite_or_none(abs(outcome_coef[idx]))
        rows.append(row)
    return rows


def _top_phrase_feature_rows(
    features: np.ndarray,
    *,
    top_n: int,
    treatment_coef: np.ndarray,
    outcome_coef: np.ndarray,
    pseudo_target_coef: np.ndarray,
    confounder_score: np.ndarray,
) -> List[Dict[str, Any]]:
    """Return agent-facing phrase evidence from 2-4 token n-grams.

    The predictive models can still use unigrams. This summary gives the
    proposal agent a phrase-biased view that is easier to map to extractable
    clinical variables.
    """
    if len(features) == 0:
        return []

    phrase_indices = [
        idx for idx, feature in enumerate(features) if 2 <= _feature_token_count(str(feature)) <= 4
    ]
    if not phrase_indices:
        return []

    phrase_indices_array = np.asarray(phrase_indices, dtype=int)
    treatment_abs = np.abs(treatment_coef)
    outcome_abs = np.abs(outcome_coef)
    pseudo_abs = np.abs(pseudo_target_coef)

    combined_score = np.maximum.reduce(
        [
            _scale_scores_for_phrase_ranking(treatment_abs),
            _scale_scores_for_phrase_ranking(outcome_abs),
            _scale_scores_for_phrase_ranking(pseudo_abs),
            _scale_scores_for_phrase_ranking(confounder_score),
        ]
    )
    order = phrase_indices_array[np.argsort(combined_score[phrase_indices_array])[::-1]]

    rows: List[Dict[str, Any]] = []
    for idx in order[:top_n]:
        row = {
            "feature": str(features[idx]),
            "token_count": int(_feature_token_count(str(features[idx]))),
            "combined_score": _finite_or_none(combined_score[idx]),
            "confounder_overlap_score": _finite_or_none(confounder_score[idx]),
            "treatment_score": _finite_or_none(treatment_coef[idx]),
            "abs_treatment_score": _finite_or_none(treatment_abs[idx]),
            "outcome_score": _finite_or_none(outcome_coef[idx]),
            "abs_outcome_score": _finite_or_none(outcome_abs[idx]),
            "pseudo_target_score": _finite_or_none(pseudo_target_coef[idx]),
            "abs_pseudo_target_score": _finite_or_none(pseudo_abs[idx]),
        }
        rows.append(row)
    return rows


def _feature_token_count(feature: str) -> int:
    return len([token for token in str(feature).split() if token])


def _scale_scores_for_phrase_ranking(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    max_abs = float(np.nanmax(np.abs(values))) if len(values) else 0.0
    if not np.isfinite(max_abs) or max_abs <= 0.0:
        return np.zeros_like(values, dtype=float)
    return np.abs(values) / max_abs


def _model_feature_scores(model: Any, n_features: int) -> np.ndarray:
    coef = getattr(model, "coef_", None)
    if coef is not None:
        values = np.asarray(coef, dtype=float).ravel()
        return _resize_scores(values, n_features)
    importances = getattr(model, "feature_importances_", None)
    if importances is not None:
        values = np.asarray(importances, dtype=float).ravel()
        return _resize_scores(values, n_features)
    booster = getattr(model, "get_booster", None)
    if booster is not None:
        try:
            score = booster().get_score(importance_type="gain")
            values = np.zeros(n_features, dtype=float)
            for key, value in score.items():
                if key.startswith("f"):
                    index = int(key[1:])
                    if 0 <= index < n_features:
                        values[index] = float(value)
            return values
        except Exception:
            pass
    return np.zeros(n_features, dtype=float)


def _resize_scores(values: np.ndarray, n_features: int) -> np.ndarray:
    values = np.asarray(values, dtype=float).ravel()
    if len(values) == n_features:
        return values
    resized = np.zeros(n_features, dtype=float)
    limit = min(n_features, len(values))
    resized[:limit] = values[:limit]
    return resized


def _merge_proposals(
    left: AgenticFeatureProposal,
    right: AgenticFeatureProposal,
) -> AgenticFeatureProposal:
    categories = _merge_ordered_values(left.categories, right.categories) or None
    feature_type = (
        "categorical"
        if left.type == "categorical" or right.type == "categorical" or categories
        else (left.type or right.type)
    )
    return AgenticFeatureProposal(
        action="add",
        name=left.name,
        type=feature_type,
        categories=categories,
        description=_merge_text_values(left.description, right.description),
        roles=_merge_ordered_values(left.roles, right.roles),
        rationale=_merge_text_values(left.rationale, right.rationale),
        expected_signal=_merge_text_values(left.expected_signal, right.expected_signal),
    )


def _merge_ordered_values(left: Any, right: Any) -> List[str]:
    values: List[str] = []
    for item in _as_list(left) + _as_list(right):
        text = str(item).strip()
        if text and text not in values:
            values.append(text)
    return values


def _merge_value_aliases(left: Any, right: Any) -> Optional[Dict[str, List[str]]]:
    merged: Dict[str, List[str]] = {}
    for source in [left, right]:
        if not isinstance(source, dict):
            continue
        for category, aliases in source.items():
            category_text = str(category).strip()
            if not category_text:
                continue
            merged[category_text] = _merge_ordered_values(
                merged.get(category_text, []),
                aliases,
            )
    return merged or None


def _merge_text_values(left: Any, right: Any) -> Optional[str]:
    left_text = str(left).strip() if left is not None else ""
    right_text = str(right).strip() if right is not None else ""
    if not left_text:
        return right_text or None
    if not right_text or right_text == left_text:
        return left_text
    return f"{left_text} / {right_text}"


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _normalize_spec(spec: ExplicitFeatureSpec) -> ExplicitFeatureSpec:
    normalized_name = _normalize_feature_name(spec.name)
    if normalized_name == spec.name:
        return spec
    return ExplicitFeatureSpec(
        name=normalized_name,
        type=spec.type,
        categories=spec.categories,
        description=spec.description,
        value_aliases=getattr(spec, "value_aliases", None),
        roles=spec.roles,
    )


def _dedupe_specs(specs: Sequence[ExplicitFeatureSpec]) -> List[ExplicitFeatureSpec]:
    by_name: Dict[str, ExplicitFeatureSpec] = {}
    for spec in specs:
        spec = _normalize_spec(spec)
        name = _normalize_feature_name(spec.name)
        if not name:
            continue
        if name not in by_name:
            by_name[name] = spec
            continue
        existing = by_name[name]
        roles = list(dict.fromkeys([*existing.roles, *spec.roles]))
        categories = _merge_ordered_values(existing.categories, spec.categories) or None
        value_aliases = _merge_value_aliases(
            getattr(existing, "value_aliases", None),
            getattr(spec, "value_aliases", None),
        )
        if existing.type == "categorical" or spec.type == "categorical" or categories:
            feature_type = "categorical"
        else:
            feature_type = existing.type
        by_name[name] = ExplicitFeatureSpec(
            name=name,
            type=feature_type,
            categories=categories,
            description=_merge_text_values(existing.description, spec.description),
            value_aliases=value_aliases,
            roles=roles,
        )
    return list(by_name.values())


def _finite_or_none(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _bow_view_to_dict(view: BoWViewConfig) -> Dict[str, Any]:
    if type(view) is not BoWViewConfig:
        raise TypeError("BoW view serialization requires BoWViewConfig")
    return asdict(view)


def _build_evidence_digest_agent_context(
    *,
    outer_fold: int,
    feature_discovery_methods: Sequence[str],
    max_proposals: int,
    clinical_question: str,
    treatment_column: str,
    outcome_column: str,
    outcome_type: str,
    current_features: Sequence[Dict[str, Any]],
    metrics: Dict[str, Any],
    importance: Dict[str, Any],
    clinical_text_examples: Sequence[str],
    embedding_evidence: Optional[Dict[str, Any]] = None,
    htr_evidence: Optional[Dict[str, Any]] = None,
    handoff_provenance: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the compact role-grouped handoff used by default for agents."""
    context: Dict[str, Any] = {
        "prompt_version": _EVIDENCE_DIGEST_PROMPT_VERSION,
        "prompt_mode": "evidence_digest",
        "outer_fold": int(outer_fold),
        "feature_discovery_methods": list(feature_discovery_methods),
        "max_proposals": int(max_proposals),
        "max_proposals_per_role": max(1, (int(max_proposals) + 1) // 2),
        "clinical_question": clinical_question,
        "estimand": {
            "treatment_column": treatment_column,
            "outcome_column": outcome_column,
            "outcome_type": outcome_type,
        },
        "instructions": [
            "You are given noisy evidence blurbs from Stage 1 text models, not a full model report.",
            "Infer explicit pre-treatment patient-level variables reflected in the blurbs and chunks.",
            "Prefer broad recall: downstream extraction, consistency checks, and causal-forest validation will prune weak variables.",
            "Do not propose raw tokens, patient names, identifiers, document-structure fields, or post-treatment variables.",
            "A role-specific prompt will be created from each evidence_digest role section.",
        ],
        "current_features": list(current_features),
        # These are deliberately complete selected notes.  The selector limits
        # the number of examples, not their contents; dropping this field in
        # the simplified evidence-digest path left the proposal agent with no
        # way to ground a readable evidence blurb back in clinical language.
        "clinical_text_examples": [str(text) for text in clinical_text_examples],
        "model_diagnostics": _agent_visible_metrics(metrics),
        "evidence_digest": _build_role_grouped_evidence_digest(
            feature_discovery_methods=feature_discovery_methods,
            importance=importance,
            embedding_evidence=embedding_evidence or {},
            htr_evidence=htr_evidence or {},
        ),
        "response_contract": _evidence_digest_response_contract(),
    }
    if handoff_provenance:
        context["handoff_provenance"] = handoff_provenance
    return context


def _evidence_digest_context_from_rich_context(context: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(context, dict):
        context = {}
    return _build_evidence_digest_agent_context(
        outer_fold=int(context.get("outer_fold") or -1),
        feature_discovery_methods=context.get("feature_discovery_methods") or [],
        max_proposals=int(context.get("max_proposals") or 1),
        clinical_question=str(context.get("clinical_question") or ""),
        treatment_column=str((context.get("estimand") or {}).get("treatment_column") or ""),
        outcome_column=str((context.get("estimand") or {}).get("outcome_column") or ""),
        outcome_type=str((context.get("estimand") or {}).get("outcome_type") or "binary"),
        current_features=context.get("current_features") or [],
        metrics=context.get("model_diagnostics") or {},
        importance=context.get("feature_importance") or {},
        clinical_text_examples=[],
        embedding_evidence=context.get("embedding_contrast_evidence") or {},
        htr_evidence=context.get("htr_attention_evidence") or {},
        handoff_provenance={
            **(context.get("handoff_provenance") or {}),
            "converted_from_rich_context": True,
        },
    )


def _build_role_grouped_evidence_digest(
    *,
    feature_discovery_methods: Sequence[str],
    importance: Dict[str, Any],
    embedding_evidence: Dict[str, Any],
    htr_evidence: Dict[str, Any],
) -> Dict[str, Any]:
    enabled = {str(method).strip().lower() for method in feature_discovery_methods}
    digest: Dict[str, Any] = {
        "confounders": {
            "role": "confounder",
            "role_definition": (
                "Variables that appear predictive of treatment assignment and baseline outcome risk."
            ),
        },
        "effect_modifiers": {
            "role": "effect_modifier",
            "role_definition": (
                "Variables that appear predictive of treatment-effect heterogeneity, "
                "R-stage pseudo-targets, residual treatment/outcome interactions, or matched-pair uplift."
            ),
        },
        "prompt_compaction": {},
    }
    for role_key, role in (
        ("confounders", "confounder"),
        ("effect_modifiers", "effect_modifier"),
    ):
        section = digest[role_key]
        if "bow" in enabled:
            section["bow_blurbs"] = _bow_evidence_digest_groups(importance, role=role)
        if "embedding_contrast" in enabled:
            section["embedding_chunks"] = _embedding_evidence_digest_groups(
                embedding_evidence,
                role=role,
            )
        if "htr" in enabled:
            section["htr_blurbs"] = _htr_evidence_digest_groups(
                htr_evidence,
                role=role,
            )
    compaction = digest["prompt_compaction"]
    if "bow" in enabled:
        compaction["bow_rows_per_list"] = _EVIDENCE_DIGEST_BOW_ROWS_PER_LIST
    if "embedding_contrast" in enabled:
        compaction.update(
            {
                "embedding_chunks_per_tail": _AGENT_PROMPT_EMBEDDING_CHUNKS_PER_TAIL,
                "embedding_chunk_text_chars": _AGENT_PROMPT_EMBEDDING_CHUNK_CHARS,
            }
        )
    if "htr" in enabled:
        compaction.update(
            {
                "htr_rows_per_stage": _AGENT_PROMPT_HTR_ROWS_PER_STAGE,
                "htr_snippet_chars": _AGENT_PROMPT_HTR_SNIPPET_CHARS,
            }
        )
    return digest


def _bow_evidence_digest_groups(
    importance: Dict[str, Any],
    *,
    role: str,
    source_prefix: str = "",
) -> List[Dict[str, Any]]:
    if not isinstance(importance, dict):
        return []
    groups: List[Dict[str, Any]] = []

    role_keys = _bow_digest_feature_keys(role)
    for view in importance.get("views", []) or []:
        if not isinstance(view, dict):
            continue
        view_name = str(view.get("view_name") or view.get("view_index") or "view")
        for key in role_keys:
            rows = _compact_feature_rows(
                view.get(key, []) or [],
                _EVIDENCE_DIGEST_BOW_ROWS_PER_LIST,
            )
            if not rows:
                continue
            groups.append(
                {
                    "source": f"{source_prefix}{view_name}.{key}",
                    "view_name": view.get("view_name"),
                    "bow_model": (
                        (view.get("view_config") or {}).get("bow_model")
                        if isinstance(view.get("view_config"), dict)
                        else None
                    )
                    or view.get("bow_model"),
                    "evidence_type": key,
                    "meaning": _bow_digest_key_description(key, role),
                    "rows": rows,
                }
            )

    if role == "effect_modifier":
        for nested_key in ["ensemble_r", "matched_pair_uplift"]:
            nested = importance.get(nested_key)
            if isinstance(nested, dict):
                groups.extend(
                    _bow_evidence_digest_groups(
                        nested,
                        role=role,
                        source_prefix=f"{source_prefix}{nested_key}.",
                    )
                )
    return groups


def _bow_digest_feature_keys(role: str) -> List[str]:
    if role == "confounder":
        return [
            "confounder_overlap",
            "treatment_positive",
            "treatment_negative",
            "outcome_positive",
            "outcome_negative",
        ]
    return [
        "pseudo_target_positive",
        "pseudo_target_negative",
        "uplift_pair_features",
        "uplift_delta_logit_positive",
        "uplift_delta_logit_negative",
        "ridge_delta_probability_positive",
        "ridge_delta_probability_negative",
    ]


def _bow_digest_key_description(key: str, role: str) -> str:
    descriptions = {
        "confounder_overlap": "Terms predictive of both treatment and outcome nuisance models.",
        "treatment_positive": "Terms positively associated with treatment assignment.",
        "treatment_negative": "Terms negatively associated with treatment assignment.",
        "outcome_positive": "Terms positively associated with outcome risk.",
        "outcome_negative": "Terms negatively associated with outcome risk.",
        "pseudo_target_positive": "Terms positively associated with the R-stage pseudo-target.",
        "pseudo_target_negative": "Terms negatively associated with the R-stage pseudo-target.",
        "uplift_pair_features": "Matched-pair uplift terms from paired treated/control patients.",
        "uplift_delta_logit_positive": "Terms increasing matched-pair treated outcome delta logit.",
        "uplift_delta_logit_negative": "Terms decreasing matched-pair treated outcome delta logit.",
        "ridge_delta_probability_positive": "Terms increasing matched-pair treated outcome delta probability.",
        "ridge_delta_probability_negative": "Terms decreasing matched-pair treated outcome delta probability.",
    }
    return descriptions.get(key, f"Top BoW evidence for {role}.")


def _embedding_evidence_digest_groups(
    evidence: Dict[str, Any],
    *,
    role: str,
) -> List[Dict[str, Any]]:
    if not isinstance(evidence, dict):
        return []
    compact = _compact_embedding_contrast_evidence(evidence)
    groups: List[Dict[str, Any]] = []
    for contrast in compact.get("contrasts", []) or []:
        if not isinstance(contrast, dict):
            continue
        if not _embedding_contrast_matches_role(contrast, role):
            continue
        item = {
            key: contrast.get(key)
            for key in [
                "name",
                "positive_label",
                "negative_label",
                "role_hint",
                "contrast_family",
                "probe_auc",
                "direction_source",
                "score_formula",
            ]
            if key in contrast
        }
        for chunk_key in [
            "positive_aligned_chunks",
            "negative_aligned_chunks",
            "positive_external_chunks",
            "negative_external_chunks",
        ]:
            chunks = contrast.get(chunk_key) or []
            if chunks:
                item[chunk_key] = chunks
        concept_scores = contrast.get("concept_probe_scores") or []
        if concept_scores:
            item["concept_probe_scores"] = concept_scores
        retrieval_terms = contrast.get("tfidf_retrieval_terms") or []
        if retrieval_terms:
            item["tfidf_retrieval_terms"] = retrieval_terms
        groups.append(item)
        if len(groups) >= _EVIDENCE_DIGEST_EMBEDDING_CONTRASTS_PER_ROLE:
            break
    return groups


def _embedding_contrast_matches_role(contrast: Dict[str, Any], role: str) -> bool:
    text = " ".join(
        str(contrast.get(key) or "").lower()
        for key in ["name", "role_hint", "contrast_family", "direction_source", "score_formula"]
    )
    if role == "confounder":
        if "effect" in text or "modifier" in text or "interaction" in text:
            return False
        return any(token in text for token in ["confounder", "treatment", "outcome"])
    return any(
        token in text
        for token in [
            "effect",
            "modifier",
            "interaction",
            "pseudo",
            "r-score",
            "r_score",
            "orthogonal",
            "residual",
            "uplift",
        ]
    )


def _htr_evidence_digest_groups(
    evidence: Dict[str, Any],
    *,
    role: str,
) -> List[Dict[str, Any]]:
    if not isinstance(evidence, dict):
        return []
    compact = _compact_htr_attention_evidence(evidence)
    stage_keys = ["nuisance"] if role == "confounder" else ["effect", "pair_uplift"]
    groups = []
    for stage_key in stage_keys:
        stage = compact.get(stage_key)
        if not isinstance(stage, dict):
            continue
        rows = stage.get("attention") or []
        if not rows:
            continue
        groups.append(
            {
                "stage": stage_key,
                "meaning": _htr_digest_stage_description(stage_key),
                "metrics": stage.get("metrics") or {},
                "rows": rows,
            }
        )
    return groups


def _htr_digest_stage_description(stage_key: str) -> str:
    if stage_key == "nuisance":
        return "HTR nuisance-model attention for treatment assignment and baseline outcome risk."
    if stage_key == "pair_uplift":
        return "HTR matched-pair uplift attention for paired treated/control outcome delta prediction."
    return "HTR R-stage/effect-model attention for treatment-effect heterogeneity."


def _evidence_digest_response_contract() -> Dict[str, Any]:
    return {
        "proposals": [
            {
                "action": "add",
                "name": "snake_case_variable_name",
                "type": "categorical|continuous",
                "categories": ["category_a", "category_b"],
                "roles": ["confounder|effect_modifier"],
                "description": "exact pre-treatment extraction target",
                "rationale": "which noisy blurbs/chunks support this variable",
                "expected_signal": "treatment/outcome or modifier signal expected",
            }
        ]
    }


def _evidence_digest_role_context(
    context: Dict[str, Any],
    *,
    role: str,
    max_proposals: int,
) -> Dict[str, Any]:
    role_key = "confounders" if role == "confounder" else "effect_modifiers"
    digest = context.get("evidence_digest") or {}
    role_evidence = digest.get(role_key) or {}
    return {
        "prompt_version": _EVIDENCE_DIGEST_ROLE_PROMPT_VERSION,
        "prompt_mode": "evidence_digest",
        "source_prompt_version": context.get("prompt_version"),
        "outer_fold": context.get("outer_fold"),
        "target_role": role,
        "max_proposals": int(max_proposals),
        "clinical_text_examples": [
            str(text) for text in context.get("clinical_text_examples", [])
        ],
        "text_blurbs": _evidence_digest_text_blurbs(role_evidence),
        "response_contract": {
            "proposals": [
                {
                    "action": "add",
                    "name": "snake_case_variable_name",
                    "type": "categorical|continuous",
                    "categories": ["category_a", "category_b"],
                    "description": "exact pre-treatment extraction target",
                    "rationale": "which blurbs support this concept",
                    "expected_signal": "brief description of the repeated text pattern",
                }
            ]
        },
        "handoff_provenance": context.get("handoff_provenance", {}),
    }


def _evidence_digest_text_blurbs(role_evidence: Dict[str, Any]) -> List[str]:
    bow_blurbs: List[str] = []
    embedding_blurbs: List[str] = []
    htr_blurbs: List[str] = []
    if not isinstance(role_evidence, dict):
        return []

    for group in role_evidence.get("bow_blurbs", []) or []:
        if not isinstance(group, dict):
            continue
        row_texts = [
            _feature_row_blurb_text(row)
            for row in group.get("rows", []) or []
            if isinstance(row, dict)
        ]
        row_texts = [text for text in row_texts if text]
        if row_texts:
            bow_blurbs.append(
                _clip_text(
                    "; ".join(row_texts),
                    _EVIDENCE_DIGEST_TEXT_BLURB_CHARS,
                )
            )

    for contrast in role_evidence.get("embedding_chunks", []) or []:
        if not isinstance(contrast, dict):
            continue
        for chunk_key in [
            "positive_aligned_chunks",
            "negative_aligned_chunks",
            "positive_external_chunks",
            "negative_external_chunks",
        ]:
            for row in contrast.get(chunk_key, []) or []:
                if not isinstance(row, dict):
                    continue
                text = _clip_text(row.get("text"), _EVIDENCE_DIGEST_TEXT_BLURB_CHARS)
                if text:
                    embedding_blurbs.append(text)

    for group in role_evidence.get("htr_blurbs", []) or []:
        if not isinstance(group, dict):
            continue
        for row in group.get("rows", []) or []:
            if not isinstance(row, dict):
                continue
            token_text = _htr_attended_token_text(row)
            if token_text:
                htr_blurbs.append(token_text)

    return _interleave_deduped_blurbs(
        [bow_blurbs, htr_blurbs, embedding_blurbs],
        limit=_EVIDENCE_DIGEST_TEXT_BLURBS_MAX,
    )


def _interleave_deduped_blurbs(
    groups: Sequence[Sequence[str]],
    *,
    limit: int,
) -> List[str]:
    deduped: List[str] = []
    seen = set()
    max_len = max((len(group) for group in groups), default=0)
    for index in range(max_len):
        for group in groups:
            if index >= len(group):
                continue
            blurb = str(group[index]).strip()
            if not blurb:
                continue
            normalized = blurb.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(blurb)
            if len(deduped) >= max(0, int(limit)):
                return deduped
    return deduped


def _feature_row_blurb_text(row: Dict[str, Any]) -> str:
    feature = str(row.get("feature") or "").strip()
    return feature


def _htr_attended_token_text(row: Dict[str, Any]) -> str:
    spans = row.get("top_token_spans")
    tokens: List[str] = []
    if isinstance(spans, list):
        for span in spans:
            if not isinstance(span, dict):
                continue
            text = str(span.get("text") or span.get("token") or span.get("span") or "").strip()
            if text and text not in tokens:
                tokens.append(text)
    if tokens:
        return _clip_text("; ".join(tokens), _EVIDENCE_DIGEST_TEXT_BLURB_CHARS)
    summary = _compact_htr_token_summary(row.get("attended_token_summary"))
    if summary:
        return summary
    return _compact_htr_token_summary(row.get("evidence_snippet"))


def _compact_htr_token_summary(value: Any) -> str:
    text = str(value or "")
    if not text.strip():
        return ""
    normalized = _normalize_text(text)
    numeric_phrases = re.findall(
        r"\b\d+(?:\.\d+)?\s*(?:cm|mm|mg/dl|g/dl|k/ul|u/l|%|ml/min|years?|y/o)\b",
        normalized,
    )
    lexical_tokens = re.findall(r"\b[a-z][a-z0-9+/-]{2,}\b", normalized)
    stop_words = set(ENGLISH_STOP_WORDS) | {
        "patient",
        "report",
        "date",
        "timepoint",
        "record",
        "note",
        "section",
        "redacted",
        "provider",
        "signature",
        "prepared",
        "mrn",
        "specimen",
        "procedure",
        "clinical",
        "history",
        "findings",
        "impression",
    }
    tokens: List[str] = []
    for token in [*numeric_phrases, *lexical_tokens]:
        token = token.strip(" -_.,;:()[]{}")
        if not token or token in stop_words or len(token) < 3:
            continue
        if token not in tokens:
            tokens.append(token)
        if len(tokens) >= 24:
            break
    return _clip_text("; ".join(tokens), _AGENT_PROMPT_HTR_SUMMARY_CHARS)


def _role_forced_raw_proposals(raw_proposals: Any, role: str) -> List[Dict[str, Any]]:
    if isinstance(raw_proposals, dict) and isinstance(raw_proposals.get("proposals"), list):
        raw_items = raw_proposals.get("proposals") or []
    elif isinstance(raw_proposals, list):
        raw_items = raw_proposals
    else:
        raw_items = []
    coerced: List[Dict[str, Any]] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        item = copy.deepcopy(raw)
        if str(item.get("action", "add")).strip().lower() == "add":
            item["roles"] = [role]
        coerced.append(item)
    return coerced


def _compact_multi_model_agent_context(context: Dict[str, Any]) -> Dict[str, Any]:
    compact = dict(context)
    if isinstance(context.get("feature_importance"), dict):
        compact["feature_importance"] = _compact_multi_model_importance(
            context["feature_importance"]
        )
    if isinstance(context.get("embedding_contrast_evidence"), dict):
        compact["embedding_contrast_evidence"] = _compact_embedding_contrast_evidence(
            context["embedding_contrast_evidence"]
        )
    if isinstance(context.get("htr_attention_evidence"), dict):
        compact["htr_attention_evidence"] = _compact_htr_attention_evidence(
            context["htr_attention_evidence"]
        )
    compact["prompt_compaction"] = {
        "feature_importance": (
            f"per-view feature lists capped at {_AGENT_PROMPT_VIEW_TOP_N}; "
            f"consensus capped at {_AGENT_PROMPT_CONSENSUS_TOP_N}"
        ),
        "embedding_contrast_evidence": (
            f"retrieved chunks capped at {_AGENT_PROMPT_EMBEDDING_CHUNKS_PER_TAIL} "
            f"per tail and {_AGENT_PROMPT_EMBEDDING_CHUNK_CHARS} chars each"
        ),
        "htr_attention_evidence": (
            f"attention rows capped at {_AGENT_PROMPT_HTR_ROWS_PER_STAGE} per stage; "
            f"snippets capped at {_AGENT_PROMPT_HTR_SNIPPET_CHARS} chars"
        ),
    }
    return compact


def _compact_multi_model_importance(importance: Dict[str, Any]) -> Dict[str, Any]:
    consensus = _compact_feature_rows(
        importance.get("phrase_consensus") or importance.get("phrase_features") or [],
        _AGENT_PROMPT_CONSENSUS_TOP_N,
    )
    compact_views = []
    for view in importance.get("views", []) or []:
        if not isinstance(view, dict):
            continue
        compact_view: Dict[str, Any] = {
            "view_name": view.get("view_name"),
            "view_index": view.get("view_index"),
            "view_config": view.get("view_config"),
            "metrics": view.get("metrics"),
            "n_features": view.get("n_features"),
            "n_bow_features": view.get("n_bow_features"),
            "n_prespecified_features": view.get("n_prespecified_features"),
            "n_prespecified_raw_features": view.get("n_prespecified_raw_features"),
            "prespecified_raw_feature_names": _clip_list(
                view.get("prespecified_raw_feature_names", []),
                50,
            ),
        }
        for key in [
            "phrase_features",
            "confounder_overlap",
            "treatment_positive",
            "treatment_negative",
            "outcome_positive",
            "outcome_negative",
            "pseudo_target_positive",
            "pseudo_target_negative",
            "uplift_pair_features",
            "uplift_delta_logit_positive",
            "uplift_delta_logit_negative",
            "ridge_delta_probability_positive",
            "ridge_delta_probability_negative",
        ]:
            compact_view[key] = _compact_feature_rows(
                view.get(key, []) or [],
                _AGENT_PROMPT_VIEW_TOP_N,
            )
        compact_views.append(compact_view)

    compact_importance = {
        "n_views": importance.get("n_views", len(compact_views)),
        "views": compact_views,
        "phrase_features": consensus,
        "phrase_consensus": consensus,
        "prompt_compaction": {
            "consensus_top_n": _AGENT_PROMPT_CONSENSUS_TOP_N,
            "per_view_list_top_n": _AGENT_PROMPT_VIEW_TOP_N,
        },
    }
    if isinstance(importance.get("ensemble_r"), dict):
        compact_importance["ensemble_r"] = _compact_multi_model_importance(importance["ensemble_r"])
    if isinstance(importance.get("matched_pair_uplift"), dict):
        compact_importance["matched_pair_uplift"] = _compact_multi_model_importance(
            importance["matched_pair_uplift"]
        )
    for key in [
        "feature_discovery_methods",
        "target_source",
        "pseudo_target_construction",
        "pair_uplift_construction",
        "nuisance_sources",
    ]:
        if key in importance:
            compact_importance[key] = importance[key]
    return compact_importance


def _compact_embedding_contrast_evidence(evidence: Dict[str, Any]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {
        key: evidence.get(key)
        for key in [
            "enabled",
            "model_name",
            "unit",
            "chunking",
            "residualized_columns",
            "external_corpora",
            "cluster_contrast_vectors",
            "n_patients",
            "n_concept_phrases",
            "skipped",
            "error",
            "disabled_reason",
        ]
        if key in evidence
    }
    contrasts = []
    for contrast in evidence.get("contrasts", []) or []:
        if not isinstance(contrast, dict):
            continue
        compact_contrast = {
            key: _round_floats(contrast.get(key))
            for key in [
                "name",
                "positive_label",
                "negative_label",
                "role_hint",
                "contrast_family",
                "n_positive",
                "n_negative",
                "mean_difference_norm",
                "probe_auc",
                "min_probe_auc",
                "direction_source",
                "direction_formula",
                "score_formula",
                "probe_auc_role",
                "direction_norm",
                "raw_interaction_norm",
                "residualized_direction_norm",
                "projection_basis",
                "treatment_direction_norm",
                "outcome_direction_norm",
                "component_cosine",
                "treatment_direction_cosine_before_residualization",
                "outcome_direction_cosine_before_residualization",
                "retrieval_skipped",
                "component_counts",
                "positive_cell_labels",
                "negative_cell_labels",
                "local_contrast_count",
                "cluster_component_index",
                "cluster_component_singular_value",
                "cluster_component_explained_energy",
                "cluster_component_loadings",
            ]
            if key in contrast
        }
        compact_contrast["positive_aligned_chunks"] = _compact_embedding_chunks(
            contrast.get("positive_aligned_chunks", []) or []
        )
        compact_contrast["negative_aligned_chunks"] = _compact_embedding_chunks(
            contrast.get("negative_aligned_chunks", []) or []
        )
        compact_contrast["positive_external_chunks"] = _compact_embedding_chunks(
            contrast.get("positive_external_chunks", []) or []
        )
        compact_contrast["negative_external_chunks"] = _compact_embedding_chunks(
            contrast.get("negative_external_chunks", []) or []
        )
        compact_contrast["concept_probe_scores"] = _compact_concept_scores(
            contrast.get("concept_probe_scores", []) or []
        )
        retrieval_terms = list(contrast.get("tfidf_retrieval_terms", []) or [])
        compact_retrieval_terms = [
            {
                key: _round_floats(row.get(key))
                for key in (
                    "term",
                    "polarity",
                    "tfidf_contrast",
                    "positive_mean_tfidf",
                    "negative_mean_tfidf",
                    "document_frequency",
                )
                if key in row
            }
            for polarity in ("positive", "negative", "unsigned")
            for row in [
                item
                for item in retrieval_terms
                if isinstance(item, dict)
                and str(item.get("polarity") or "unsigned") == polarity
            ][:_AGENT_PROMPT_CONCEPT_TOP_N]
            if isinstance(row, dict)
        ]
        compact_contrast["tfidf_retrieval_terms"] = compact_retrieval_terms
        if "tfidf_retrieval" in contrast:
            compact_contrast["tfidf_retrieval"] = _round_floats(
                contrast.get("tfidf_retrieval")
            )
        contrasts.append(compact_contrast)
    compact["contrasts"] = contrasts
    compact["prompt_compaction"] = {
        "chunks_per_tail": _AGENT_PROMPT_EMBEDDING_CHUNKS_PER_TAIL,
        "chunk_text_chars": _AGENT_PROMPT_EMBEDDING_CHUNK_CHARS,
        "concept_top_n": _AGENT_PROMPT_CONCEPT_TOP_N,
    }
    return compact


def _compact_htr_attention_evidence(evidence: Dict[str, Any]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    for stage_key in ["nuisance", "effect", "pair_uplift"]:
        stage_evidence = evidence.get(stage_key)
        if not isinstance(stage_evidence, dict):
            continue
        compact[stage_key] = {
            "metrics": _round_floats(stage_evidence.get("metrics", {})),
            "attention": _compact_htr_attention_rows(
                stage_evidence.get("attention", []) or [],
                max_rows=_AGENT_PROMPT_HTR_ROWS_PER_STAGE,
            ),
        }
    if compact:
        compact["prompt_compaction"] = {
            "rows_per_stage": _AGENT_PROMPT_HTR_ROWS_PER_STAGE,
            "snippet_chars": _AGENT_PROMPT_HTR_SNIPPET_CHARS,
            "summary_chars": _AGENT_PROMPT_HTR_SUMMARY_CHARS,
        }
    return compact


def _compact_htr_attention_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    max_rows: int,
) -> List[Dict[str, Any]]:
    compact: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        already_compact_text = bool(
            row.get("evidence_snippet")
            or row.get("top_token_spans")
            or row.get("attended_token_summary")
        )
        if not _attention_row_has_usable_text(row) and not already_compact_text:
            continue
        spans = _parse_top_token_spans(
            row.get("top_token_spans") or row.get("top_token_spans_json")
        )
        item: Dict[str, Any] = {}
        for key in [
            "row_id",
            "_oci_row_id",
            "outer_fold",
            "fold",
            "stage",
            "model_family",
            "chunk_index",
            "effect_objective",
            "target_source",
            "view_name",
            "pair_side",
            "candidate_row_id",
            "control_row_id",
        ]:
            if key in row:
                item[key] = _round_floats(row[key])
        for key in [
            "attention",
            "attention_score",
            "chunk_attention",
            "e_hat",
            "m_hat",
            "e_hat_raw",
            "m_hat_raw",
            "y_residual",
            "t_residual",
            "tau_hat_r_stage",
            "tau_logit_modifier",
            "r_pseudo_outcome",
            "r_loss",
            "effect_loss",
            "effect_loss_at_zero_tau",
            "pair_delta_logit",
            "pair_pred_prob",
            "pair_base_prob",
            "pair_score_abs_diff_sum",
        ]:
            if key in row:
                item[key] = _round_floats(row[key])
        snippet = _clip_text(row.get("evidence_snippet"), _AGENT_PROMPT_HTR_SNIPPET_CHARS)
        if not snippet:
            snippet = _attention_evidence_snippet(
                row.get("chunk_text"),
                spans,
                row.get("highlighted_chunk_text"),
            )
        if snippet:
            item["evidence_snippet"] = _clip_text(snippet, _AGENT_PROMPT_HTR_SNIPPET_CHARS)
        if spans:
            item["top_token_spans"] = _compact_token_spans(spans)
        summary = _clip_text(
            row.get("attended_token_summary"),
            _AGENT_PROMPT_HTR_SUMMARY_CHARS,
        )
        if summary:
            item["attended_token_summary"] = summary
        if item:
            compact.append(item)
        if len(compact) >= max(0, int(max_rows)):
            break
    return compact


def _redact_htr_attention_evidence(evidence: Dict[str, Any]) -> Dict[str, Any]:
    compact = _compact_htr_attention_evidence(evidence)
    for stage_key in ["nuisance", "effect", "pair_uplift"]:
        stage_evidence = compact.get(stage_key)
        if not isinstance(stage_evidence, dict):
            continue
        redacted_rows = []
        for row in stage_evidence.get("attention", []) or []:
            if not isinstance(row, dict):
                continue
            redacted = {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "evidence_snippet",
                    "top_token_spans",
                    "attended_token_summary",
                }
            }
            redacted["text_redacted"] = True
            redacted_rows.append(redacted)
        stage_evidence["attention"] = redacted_rows
    return compact


def _redact_evidence_digest(digest: Dict[str, Any]) -> Dict[str, Any]:
    redacted = copy.deepcopy(digest) if isinstance(digest, dict) else {}
    for role_key in ["confounders", "effect_modifiers"]:
        section = redacted.get(role_key)
        if not isinstance(section, dict):
            continue
        for contrast in section.get("embedding_chunks", []) or []:
            if not isinstance(contrast, dict):
                continue
            for chunk_key in [
                "positive_aligned_chunks",
                "negative_aligned_chunks",
                "positive_external_chunks",
                "negative_external_chunks",
            ]:
                for row in contrast.get(chunk_key, []) or []:
                    if not isinstance(row, dict):
                        continue
                    if "text" in row:
                        row["text"] = None
                    row["text_redacted"] = True
        for group in section.get("htr_blurbs", []) or []:
            if not isinstance(group, dict):
                continue
            for row in group.get("rows", []) or []:
                if not isinstance(row, dict):
                    continue
                row.pop("evidence_snippet", None)
                row.pop("top_token_spans", None)
                row.pop("attended_token_summary", None)
                row["text_redacted"] = True
    return redacted


def _compact_feature_rows(rows: Sequence[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
    compact = []
    for row in list(rows)[: max(0, int(top_n))]:
        if not isinstance(row, dict):
            continue
        compact.append(
            {
                key: _round_floats(value)
                for key, value in row.items()
                if key
                in {
                    "feature",
                    "token_count",
                    "score",
                    "combined_score",
                    "confounder_overlap_score",
                    "treatment_score",
                    "abs_treatment_score",
                    "outcome_score",
                    "abs_outcome_score",
                    "pseudo_target_score",
                    "abs_pseudo_target_score",
                    "uplift_delta_logit_score",
                    "abs_uplift_delta_logit_score",
                    "supporting_view_count",
                    "supporting_views",
                    "best_abs_confounder_score",
                    "mean_abs_confounder_score",
                    "best_abs_effect_score",
                    "mean_abs_effect_score",
                }
            }
        )
    return compact


def _compact_embedding_chunks(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    compact = []
    for row in list(rows)[:_AGENT_PROMPT_EMBEDDING_CHUNKS_PER_TAIL]:
        if not isinstance(row, dict):
            continue
        item = {
            "row_id": row.get("row_id"),
            "chunk_index": row.get("chunk_index"),
            "score": _round_floats(row.get("score")),
            "text": _clip_text(row.get("text"), _AGENT_PROMPT_EMBEDDING_CHUNK_CHARS),
        }
        for key in ["corpus", "cache_path", "row_index", "metadata"]:
            if key in row:
                item[key] = _round_floats(row.get(key))
        compact.append(item)
    return compact


def _compact_concept_scores(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    compact = []
    for row in list(rows)[:_AGENT_PROMPT_CONCEPT_TOP_N]:
        if not isinstance(row, dict):
            continue
        compact.append(
            {
                "concept": row.get("concept"),
                "score": _round_floats(row.get("score")),
            }
        )
    return compact


def _clip_list(values: Any, max_items: int) -> List[Any]:
    if not isinstance(values, list):
        return []
    return values[: max(0, int(max_items))]


def _clip_text(value: Any, max_chars: int) -> str:
    text = " ".join(str(value or "").split())
    limit = max(0, int(max_chars))
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _round_floats(value: Any) -> Any:
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return round(float(value), 5)
    if isinstance(value, np.floating):
        numeric = float(value)
        if not np.isfinite(numeric):
            return None
        return round(numeric, 5)
    if isinstance(value, dict):
        return {key: _round_floats(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_floats(item) for item in value]
    return value


def _multi_view_metrics(view_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not view_results:
        return {"n_bow_views": 0}
    primary = _select_primary_bow_view(view_results)
    primary_metrics = primary.get("metrics", {})
    metrics: Dict[str, Any] = {
        "n_bow_views": int(len(view_results)),
        "primary_view": str(primary["view"].name),
        "primary_view_index": int(primary["view_index"]),
        "views": [
            {
                "view_name": str(result.get("view_name") or result["view"].name),
                "view_index": int(result["view_index"]),
                "view_config": _bow_view_to_dict(result["view"]),
                "metrics": _agent_visible_metrics(result.get("metrics", {})),
            }
            for result in view_results
        ],
    }
    for key, value in _scalar_metrics(primary_metrics).items():
        metrics[f"primary_{key}"] = value
    best_improvement = max(
        (
            value
            for value in (
                _finite_or_none(result.get("metrics", {}).get("r_loss_relative_improvement"))
                for result in view_results
            )
            if value is not None
        ),
        default=None,
    )
    metrics["best_r_loss_relative_improvement"] = best_improvement
    return metrics


def _select_primary_bow_view(view_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    def score(result: Dict[str, Any]) -> Tuple[float, float]:
        metrics = result.get("metrics", {})
        improvement = _finite_or_none(metrics.get("r_loss_relative_improvement"))
        tau_corr = _finite_or_none(metrics.get("tau_hat_pseudo_target_corr"))
        return (
            float("-inf") if improvement is None else improvement,
            float("-inf") if tau_corr is None else abs(tau_corr),
        )

    return max(view_results, key=score)


def _multi_view_importance(
    view_results: Sequence[Dict[str, Any]],
    *,
    top_n: int,
) -> Dict[str, Any]:
    views = []
    for result in view_results:
        importance = dict(result.get("importance", {}))
        importance["view_name"] = str(result.get("view_name") or result["view"].name)
        importance["view_index"] = int(result["view_index"])
        importance["view_config"] = _bow_view_to_dict(result["view"])
        importance["metrics"] = _agent_visible_metrics(result.get("metrics", {}))
        views.append(importance)
    consensus = _consensus_phrase_feature_rows(views, top_n=top_n)
    return {
        "n_views": int(len(views)),
        "views": views,
        "phrase_features": consensus,
        "phrase_consensus": consensus,
    }


def _consensus_phrase_feature_rows(
    view_importances: Sequence[Dict[str, Any]],
    *,
    top_n: int,
) -> List[Dict[str, Any]]:
    accumulator: Dict[str, Dict[str, Any]] = {}
    for view in view_importances:
        view_name = str(view.get("view_name", "view"))
        for row in view.get("phrase_features", []) or []:
            feature = str(row.get("feature", "")).strip()
            if not feature:
                continue
            key = _normalize_text(feature)
            entry = accumulator.setdefault(
                key,
                {
                    "feature": feature,
                    "supporting_views": set(),
                    "view_scores": [],
                    "abs_confounder_scores": [],
                    "abs_effect_scores": [],
                },
            )
            entry["supporting_views"].add(view_name)
            confounder_score = abs(float(row.get("confounder_overlap_score") or 0.0))
            effect_score = abs(float(row.get("abs_pseudo_target_score") or 0.0))
            entry["abs_confounder_scores"].append(confounder_score)
            entry["abs_effect_scores"].append(effect_score)
            entry["view_scores"].append(
                {
                    "view_name": view_name,
                    "combined_score": row.get("combined_score"),
                    "confounder_overlap_score": row.get("confounder_overlap_score"),
                    "treatment_score": row.get("treatment_score"),
                    "outcome_score": row.get("outcome_score"),
                    "pseudo_target_score": row.get("pseudo_target_score"),
                }
            )

    rows: List[Dict[str, Any]] = []
    for entry in accumulator.values():
        confounder_scores = entry["abs_confounder_scores"]
        effect_scores = entry["abs_effect_scores"]
        supporting_views = sorted(entry["supporting_views"])
        best_confounder = max(confounder_scores) if confounder_scores else 0.0
        best_effect = max(effect_scores) if effect_scores else 0.0
        mean_confounder = float(np.mean(confounder_scores)) if confounder_scores else 0.0
        mean_effect = float(np.mean(effect_scores)) if effect_scores else 0.0
        rows.append(
            {
                "feature": entry["feature"],
                "supporting_view_count": int(len(supporting_views)),
                "supporting_views": supporting_views,
                "best_abs_confounder_score": _finite_or_none(best_confounder),
                "mean_abs_confounder_score": _finite_or_none(mean_confounder),
                "best_abs_effect_score": _finite_or_none(best_effect),
                "mean_abs_effect_score": _finite_or_none(mean_effect),
                "view_scores": entry["view_scores"],
            }
        )

    rows.sort(
        key=lambda row: (
            int(row["supporting_view_count"]),
            float(row.get("best_abs_confounder_score") or 0.0),
            float(row.get("best_abs_effect_score") or 0.0),
            float(row.get("mean_abs_confounder_score") or 0.0),
            float(row.get("mean_abs_effect_score") or 0.0),
        ),
        reverse=True,
    )
    return rows[:top_n]


def _agent_visible_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in metrics.items() if not _is_oracle_metric_name(key)}


def _is_oracle_metric_name(key: Any) -> bool:
    name = str(key).lower()
    return name.startswith("oracle_") or name.startswith("true_") or "true_" in name


def _scalar_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value for key, value in metrics.items() if not isinstance(value, (list, tuple, dict))
    }


def _prefix_metrics(prefix: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {f"{prefix}{key}": value for key, value in _scalar_metrics(metrics).items()}


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, default=_json_default) + "\n")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)


def _read_json(path: Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _read_csv_records(path: Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    return pd.read_csv(path).to_dict(orient="records")
