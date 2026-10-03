"""Agentic explicit-feature causal forest search.

This module runs an adaptive, LLM-guided variable search around the existing
explicit-feature causal forest. The reported performance comes from outer CV;
all feature-set decisions are made with inner CV on each outer-training split.
"""

import hashlib
import json
import logging
import math
import os
import re
import shlex
import subprocess
import tempfile
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

from ..config import (
    AgenticFeatureSearchConfig,
    AppliedInferenceConfig,
    ExplicitFeatureForestConfig,
    ExplicitFeatureSpec,
)
from ..extraction import (
    COMPLETE_PAGED_VERSION,
    CONTRACT_LEXICAL_CONTEXT_VERSION,
    EXTRACTION_GROUPING_VERSION,
    CompleteFeatureContract,
    CompletePageResponse,
    CompletePagingGeometry,
    ExtractionCache,
    VLLMFeatureExtractor,
    build_complete_paged_coverage_ledger,
    plan_complete_paged_requests,
    reconcile_complete_page_responses,
    resolve_vllm_reasoning_parser,
    strip_reasoning_trace,
)
from ..extraction.explicit_features import (
    _missing_values_for_specs,
    _parse_extraction_response_with_issues,
    build_extraction_prompt,
    build_extraction_repair_prompt,
)
from ..extraction.llm_routing import (
    OpenAIClientPool,
    call_with_exponential_backoff,
    google_json_response_format_kwargs,
    retry_delay,
)
from ..extraction.llm_routing import parse_server_urls
from ..models.causal_forest_head import CausalForestHead
from ..models.structured_interaction_head import StructuredInteractionHead
from .applied_explicit_feature_forest import _build_features, _hstack_present

logger = logging.getLogger(__name__)

AGENT_PROMPT_VERSION = "agentic_explicit_feature_search_v1"
BROAD_AGENT_PROMPT_VERSION = "agentic_explicit_feature_broad_screen_v2"
EXTRACTION_PROMPT_VERSION = "explicit_features_v5"
VALID_ACTIONS = {"add", "remove", "update_role", "none"}
VALID_ROLES = {"confounder", "effect_modifier"}
VALID_TYPES = {"categorical", "continuous"}
_CONCEPT_INVENTORY_PROMPT_VERSIONS = {
    "multi_model_agentic_concept_inventory_v1",
    "multi_model_agentic_cluster_labeling_v1",
    "multi_model_agentic_cluster_labeling_v2",
}
_PARSIMONY_FACTOR_PROMPT_VERSION = "multi_model_agentic_parsimony_factor_v1"
_AUTO_MODEL_NAME_VALUES = {"", "auto", "discover", "server"}
_MODEL_NAME_AUTODISCOVERY_ALIASES = {"Qwen/Qwen3.6-27B"}
_DASH_TRANSLATION = dict.fromkeys(
    map(ord, "\u2010\u2011\u2012\u2013\u2014\u2212"),
    "-",
)
_MISSING_VALUE_LABELS = {
    "",
    "unknown",
    "unk",
    "not_reported",
    "not reported",
    "not_assessed",
    "not assessed",
    "not_tested",
    "not tested",
    "unavailable",
    "not_available",
    "not available",
    "na",
    "n/a",
    "none",
    "null",
    "missing",
    "not_applicable",
    "not applicable",
    "indeterminate",
}


def _is_auto_model_name(model_name: Optional[str]) -> bool:
    return model_name is None or str(model_name).strip().lower() in _AUTO_MODEL_NAME_VALUES


def _should_autodiscover_model_name(model_name: Optional[str]) -> bool:
    if _is_auto_model_name(model_name):
        return True
    return str(model_name).strip() in _MODEL_NAME_AUTODISCOVERY_ALIASES


def _model_id_from_openai_model(model: Any) -> Optional[str]:
    if isinstance(model, str):
        return model
    if isinstance(model, dict):
        value = model.get("id")
        return str(value) if value else None
    value = getattr(model, "id", None)
    return str(value) if value else None


def _discover_openai_compatible_model_name(
    client: Any,
    server_url: Optional[str],
    purpose: str,
) -> str:
    try:
        response = client.models.list()
    except Exception as exc:
        raise RuntimeError(
            f"Could not autodiscover {purpose} model from {server_url or 'configured server'} "
            "via /models. Provide an explicit model name instead."
        ) from exc

    models = getattr(response, "data", response)
    for model in models or []:
        model_id = _model_id_from_openai_model(model)
        if model_id:
            logger.info(
                "Autodiscovered %s model from %s: %s",
                purpose,
                server_url or "configured server",
                model_id,
            )
            return model_id
    raise RuntimeError(
        f"Could not autodiscover {purpose} model from {server_url or 'configured server'}: "
        "/models returned no model ids. Provide an explicit model name instead."
    )


def _endpoint_model_inventory_identity(models_by_url: Dict[str, str]) -> str:
    """Return a cache/checkpoint identity for one possibly heterogeneous pool."""
    normalized = {
        str(url): str(model)
        for url, model in sorted(models_by_url.items())
        if str(url).strip() and str(model).strip()
    }
    if not normalized:
        raise ValueError("Endpoint model inventory cannot be empty")
    unique_models = sorted(set(normalized.values()))
    if len(unique_models) == 1:
        return unique_models[0]
    digest = hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return f"heterogeneous_endpoint_model_pool:{digest}"


@dataclass
class AgenticFeatureProposal:
    """Validated proposal emitted by the feature-search agent."""

    action: str
    name: str
    type: Optional[str] = None
    categories: Optional[List[str]] = None
    description: Optional[str] = None
    roles: List[str] = field(default_factory=list)
    rationale: Optional[str] = None
    expected_signal: Optional[str] = None


@dataclass
class SplitEvaluation:
    """Predictions and metrics for one train/test split."""

    predictions: pd.DataFrame
    metrics: Dict[str, Any]


@dataclass
class BroadScreenPreparedFold:
    """Inventory proposal state prepared before broad-screen extraction."""

    outer_fold: int
    train_idx: np.ndarray
    proposals: List[AgenticFeatureProposal]


def run_agentic_explicit_feature_forest(
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    output_path: Path,
    device=None,
    num_workers: int = 1,
    proposal_agent: Optional[Any] = None,
    extraction_provider: Optional[Any] = None,
    evaluator: Optional[Any] = None,
) -> None:
    """Run nested-CV agentic explicit-feature causal forest inference."""
    del device, num_workers
    runner = AgenticFeatureSearchRunner(
        dataset=dataset,
        config=config,
        output_path=output_path,
        proposal_agent=proposal_agent,
        extraction_provider=extraction_provider,
        evaluator=evaluator,
    )
    runner.run()


class AgenticFeatureSearchRunner:
    """Nested-CV runner for adaptive explicit-feature search."""

    def __init__(
        self,
        dataset: pd.DataFrame,
        config: AppliedInferenceConfig,
        output_path: Path,
        proposal_agent: Optional[Any] = None,
        extraction_provider: Optional[Any] = None,
        evaluator: Optional[Any] = None,
    ):
        self.dataset = dataset.reset_index(drop=True).copy()
        self.config = config
        self.output_path = Path(output_path)
        self.artifact_dir = self.output_path.parent / "agentic_feature_search"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        self.search_config = getattr(
            config.architecture,
            "agentic_feature_search",
            AgenticFeatureSearchConfig(),
        )
        self.cf_config = getattr(
            config.architecture,
            "explicit_feature_forest",
            ExplicitFeatureForestConfig(),
        )
        self.initial_specs = (
            list(config.explicit_features.features)
            if getattr(config.explicit_features, "enabled", False)
            else []
        )
        if config.explicit_features.features and not config.explicit_features.enabled:
            logger.info(
                "Ignoring configured explicit_features.features because "
                "explicit_features.enabled=False"
            )
        if not self.initial_specs:
            logger.info("Agentic explicit-feature search is starting from an empty feature set")

        self.proposal_agent = proposal_agent or make_feature_search_agent(self.search_config)
        self.extraction_provider = extraction_provider or make_explicit_feature_extraction_provider(
            config=config,
            output_dir=self.artifact_dir,
        )
        self.evaluator = evaluator or CausalForestExplicitEvaluator(
            config=config,
            cf_config=self.cf_config,
        )

        self.decision_events: List[Dict[str, Any]] = []
        self.inner_metric_rows: List[Dict[str, Any]] = []
        self.outer_metric_rows: List[Dict[str, Any]] = []
        self.feature_set_rows: List[Dict[str, Any]] = []
        self.screening_metric_rows: List[Dict[str, Any]] = []

    def run(self) -> None:
        """Execute outer CV, inner adaptive search, and final reporting."""
        logger.info("=" * 80)
        logger.info("AGENTIC EXPLICIT FEATURE CAUSAL FOREST")
        logger.info("=" * 80)

        # Iterative mode needs the starting variables before the first prompt
        # because the agent sees extraction summaries. Broad-screen mode
        # extracts initial variables together with the broad inventory union.
        if self.search_config.search_mode != "broad_screen":
            self.dataset = self.extraction_provider.ensure_features(
                self.dataset,
                self.initial_specs,
            )

        outer_splits = _make_splits(
            self.dataset,
            self.config,
            n_splits=self.search_config.outer_folds,
            random_state=self.search_config.random_state,
        )

        selected_by_outer_fold: Dict[int, List[ExplicitFeatureSpec]] = {}
        if self.search_config.search_mode == "broad_screen":
            selected_by_outer_fold = self._search_all_outer_trains_broad_screen(outer_splits)

        outer_predictions = []
        for outer_fold, (train_idx, test_idx) in enumerate(outer_splits, start=1):
            logger.info(
                "Outer fold %s/%s final evaluation: train=%s test=%s",
                outer_fold,
                len(outer_splits),
                len(train_idx),
                len(test_idx),
            )
            selected_specs = selected_by_outer_fold.get(outer_fold)
            if selected_specs is None:
                selected_specs = self._search_outer_train(outer_fold, train_idx)
            selected_specs = self._resolve_selected_aliases(
                outer_fold=outer_fold,
                selected_specs=selected_specs,
            )
            selected_specs = self._harmonize_value_contracts(
                outer_fold=outer_fold,
                selected_specs=selected_specs,
            )
            self.dataset = self.extraction_provider.ensure_features(self.dataset, selected_specs)

            train_df = self.dataset.iloc[train_idx].copy()
            test_df = self.dataset.iloc[test_idx].copy()
            final_eval = self.evaluator.evaluate_split(
                train_df=train_df,
                test_df=test_df,
                specs=selected_specs,
                fold_id=outer_fold,
            )
            preds = final_eval.predictions.copy()
            preds["outer_fold"] = outer_fold
            preds["selected_feature_names"] = ",".join(spec.name for spec in selected_specs)
            outer_predictions.append(preds)

            metrics = {
                "outer_fold": outer_fold,
                "stage": "outer_final",
                "n_selected_features": len(selected_specs),
                **_without_list_values(final_eval.metrics),
            }
            self.outer_metric_rows.append(metrics)
            self.feature_set_rows.append(
                {
                    "outer_fold": outer_fold,
                    "stage": "selected",
                    "features": [_spec_to_dict(spec) for spec in selected_specs],
                }
            )

        results_df = pd.concat(outer_predictions).sort_index()
        self._save_predictions(results_df)
        self._save_artifacts()

    def _search_outer_train(
        self,
        outer_fold: int,
        outer_train_idx: np.ndarray,
    ) -> List[ExplicitFeatureSpec]:
        """Run the configured feature search for one outer-training split."""
        if self.search_config.search_mode == "broad_screen":
            return self._search_outer_train_broad_screen(outer_fold, outer_train_idx)
        return self._search_outer_train_iterative(outer_fold, outer_train_idx)

    def _search_outer_train_iterative(
        self,
        outer_fold: int,
        outer_train_idx: np.ndarray,
    ) -> List[ExplicitFeatureSpec]:
        """Run the inner adaptive search for one outer-training split."""
        current_specs = list(self.initial_specs)
        accepted_additions = 0

        baseline_rows, baseline_summary = self._evaluate_inner_cv(
            outer_fold=outer_fold,
            iteration=0,
            candidate_name="initial",
            train_idx=outer_train_idx,
            specs=current_specs,
        )
        self._record_inner_rows(baseline_rows, accepted=True)

        for iteration in range(1, self.search_config.max_iterations + 1):
            context = self._build_agent_context(
                outer_fold=outer_fold,
                iteration=iteration,
                train_idx=outer_train_idx,
                current_specs=current_specs,
                current_summary=baseline_summary,
            )
            try:
                raw_proposals = self.proposal_agent.propose(context)
            except Exception as exc:
                error_payload = {
                    "error": repr(exc),
                    "context": context,
                }
                if self.search_config.save_agent_raw_output:
                    error_payload["agent_raw_output"] = _get_agent_response_trace(
                        self.proposal_agent
                    )
                self._record_decision(
                    outer_fold,
                    iteration,
                    "agent_proposal_error",
                    error_payload,
                )
                self._save_decision_events()
                raise
            proposal_payload = {
                "raw_count": len(raw_proposals),
                "valid_count": None,
                "rejected": None,
                "context": context,
                "raw_proposals": raw_proposals,
            }
            if self.search_config.save_agent_raw_output:
                proposal_payload["agent_raw_output"] = _get_agent_response_trace(
                    self.proposal_agent
                )
            proposals, rejected = validate_agentic_proposals(
                raw_proposals,
                current_specs=current_specs,
                search_config=self.search_config,
                allow_removals=accepted_additions > 0,
            )
            proposal_payload["valid_count"] = len(proposals)
            proposal_payload["rejected"] = rejected
            self._record_decision(
                outer_fold,
                iteration,
                "agent_proposals",
                proposal_payload,
            )

            if not proposals:
                logger.info("Outer fold %s iteration %s: no valid proposals", outer_fold, iteration)
                break

            candidate_results = []
            for candidate_id, proposal_group in _candidate_groups(proposals):
                candidate_specs = apply_proposals(current_specs, proposal_group)
                if _spec_names(candidate_specs) == _spec_names(current_specs):
                    continue
                self.dataset = self.extraction_provider.ensure_features(
                    self.dataset,
                    candidate_specs,
                )
                proposal_specs = _candidate_proposal_specs(
                    current_specs=current_specs,
                    candidate_specs=candidate_specs,
                    proposal_group=proposal_group,
                )
                role_diagnostics = evaluate_candidate_role_diagnostics(
                    dataset=self.dataset.iloc[outer_train_idx],
                    current_specs=current_specs,
                    candidate_specs=proposal_specs,
                    config=self.config,
                    search_config=self.search_config,
                )
                coverage_failures = _coverage_failures(
                    self.dataset.iloc[outer_train_idx],
                    proposal_specs,
                    self.search_config.min_feature_coverage,
                )
                if coverage_failures:
                    summary = {"coverage_failures": coverage_failures}
                    if role_diagnostics:
                        summary["role_diagnostics"] = role_diagnostics
                    candidate_results.append(
                        {
                            "candidate_id": candidate_id,
                            "proposal_group": proposal_group,
                            "specs": candidate_specs,
                            "rows": [],
                            "summary": summary,
                            "comparison": {
                                "passes_acceptance": False,
                                "rejection_reason": "low_feature_coverage",
                                "coverage_failures": coverage_failures,
                            },
                        }
                    )
                    continue
                rows, summary = self._evaluate_inner_cv(
                    outer_fold=outer_fold,
                    iteration=iteration,
                    candidate_name=candidate_id,
                    train_idx=outer_train_idx,
                    specs=candidate_specs,
                )
                if role_diagnostics:
                    summary = dict(summary)
                    summary["role_diagnostics"] = role_diagnostics
                comparison = compare_candidate_to_baseline(
                    baseline_rows=baseline_rows,
                    candidate_rows=rows,
                    search_config=self.search_config,
                )
                candidate_results.append(
                    {
                        "candidate_id": candidate_id,
                        "proposal_group": proposal_group,
                        "specs": candidate_specs,
                        "rows": rows,
                        "summary": summary,
                        "comparison": comparison,
                    }
                )
                self._record_inner_rows(rows, accepted=False)

            accepted = _choose_accepted_candidate(candidate_results)
            self._record_decision(
                outer_fold,
                iteration,
                "candidate_evaluations",
                [
                    {
                        "candidate_id": item["candidate_id"],
                        "proposals": [asdict(p) for p in item["proposal_group"]],
                        "summary": item["summary"],
                        "comparison": item["comparison"],
                        "accepted": accepted is item,
                    }
                    for item in candidate_results
                ],
            )

            if accepted is None:
                logger.info(
                    "Outer fold %s iteration %s: no candidate passed acceptance thresholds",
                    outer_fold,
                    iteration,
                )
                if self.search_config.stop_after_rejected_iteration:
                    break
                continue

            current_specs = accepted["specs"]
            baseline_rows = accepted["rows"]
            baseline_summary = accepted["summary"]
            accepted_additions += sum(
                1 for proposal in accepted["proposal_group"] if proposal.action == "add"
            )
            self._record_inner_rows(accepted["rows"], accepted=True)
            self.feature_set_rows.append(
                {
                    "outer_fold": outer_fold,
                    "iteration": iteration,
                    "stage": "accepted_inner",
                    "candidate_id": accepted["candidate_id"],
                    "features": [_spec_to_dict(spec) for spec in current_specs],
                }
            )
            logger.info(
                "Outer fold %s iteration %s: accepted %s",
                outer_fold,
                iteration,
                accepted["candidate_id"],
            )

        return current_specs

    def _search_outer_train_broad_screen(
        self,
        outer_fold: int,
        outer_train_idx: np.ndarray,
    ) -> List[ExplicitFeatureSpec]:
        """Run broad-screen search for one fold.

        The main run path prepares all broad-screen folds first so their proposal
        union can be extracted once. This single-fold path is retained for direct
        use and tests.
        """
        prepared = self._prepare_broad_screen_fold(outer_fold, outer_train_idx)
        union_specs = _canonicalize_broad_screen_prepared_folds(
            self.initial_specs,
            [prepared],
        )
        if union_specs:
            self.dataset = self.extraction_provider.ensure_features(
                self.dataset,
                union_specs,
            )
        return self._screen_and_refine_broad_candidates(prepared)

    def _search_all_outer_trains_broad_screen(
        self,
        outer_splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    ) -> Dict[int, List[ExplicitFeatureSpec]]:
        """Prepare all broad-screen folds, union-extract, then screen/refine."""
        prepared_folds: List[BroadScreenPreparedFold] = []
        for outer_fold, (train_idx, test_idx) in enumerate(outer_splits, start=1):
            logger.info(
                "Outer fold %s/%s broad-screen proposal prep: train=%s test=%s",
                outer_fold,
                len(outer_splits),
                len(train_idx),
                len(test_idx),
            )
            prepared = self._prepare_broad_screen_fold(outer_fold, train_idx)
            prepared_folds.append(prepared)

        union_specs = _canonicalize_broad_screen_prepared_folds(
            self.initial_specs,
            prepared_folds,
        )
        if union_specs:
            logger.info(
                "Broad-screen union extraction across outer folds: %s feature(s)",
                len(union_specs),
            )
            self.dataset = self.extraction_provider.ensure_features(
                self.dataset,
                union_specs,
            )

        selected_by_outer_fold = {}
        for prepared in prepared_folds:
            selected_by_outer_fold[prepared.outer_fold] = self._screen_and_refine_broad_candidates(
                prepared
            )
        return selected_by_outer_fold

    def _prepare_broad_screen_fold(
        self,
        outer_fold: int,
        outer_train_idx: np.ndarray,
    ) -> BroadScreenPreparedFold:
        if self.search_config.agent_max_tokens < 8000:
            logger.warning(
                "broad_screen mode asks for up to %s proposals but agent_max_tokens=%s; "
                "consider increasing agent_max_tokens to reduce truncation risk",
                self.search_config.broad_candidate_count,
                self.search_config.agent_max_tokens,
            )

        context = self._build_broad_inventory_context(
            outer_fold=outer_fold,
            train_idx=outer_train_idx,
        )

        try:
            raw_proposals = self.proposal_agent.propose(context)
        except Exception as exc:
            error_payload = {
                "error": repr(exc),
                "context": context,
            }
            if self.search_config.save_agent_raw_output:
                error_payload["agent_raw_output"] = _get_agent_response_trace(self.proposal_agent)
            self._record_decision(
                outer_fold,
                0,
                "agent_proposal_error",
                error_payload,
            )
            self._save_decision_events()
            raise

        proposal_payload = {
            "raw_count": len(raw_proposals),
            "valid_count": None,
            "rejected": None,
            "context": context,
            "raw_proposals": raw_proposals,
        }
        if self.search_config.save_agent_raw_output:
            proposal_payload["agent_raw_output"] = _get_agent_response_trace(self.proposal_agent)
        proposals, rejected = validate_agentic_proposals(
            raw_proposals,
            current_specs=list(self.initial_specs),
            search_config=self.search_config,
            allow_removals=False,
            max_additions=self.search_config.broad_candidate_count,
            allow_duplicate_additions=True,
        )
        proposal_payload["valid_count"] = len(proposals)
        proposal_payload["rejected"] = rejected
        self._record_decision(
            outer_fold,
            0,
            "agent_proposals",
            proposal_payload,
        )

        if not proposals:
            logger.info("Outer fold %s broad_screen: no valid proposals", outer_fold)
        return BroadScreenPreparedFold(
            outer_fold=outer_fold,
            train_idx=outer_train_idx,
            proposals=proposals,
        )

    def _screen_and_refine_broad_candidates(
        self,
        prepared: BroadScreenPreparedFold,
    ) -> List[ExplicitFeatureSpec]:
        """Adaptively select from pre-extracted broad candidates plus new proposals."""
        outer_fold = prepared.outer_fold
        outer_train_idx = prepared.train_idx
        current_specs = list(self.initial_specs)
        proposals = prepared.proposals

        baseline_rows, baseline_summary = self._evaluate_inner_cv(
            outer_fold=outer_fold,
            iteration=0,
            candidate_name="initial",
            train_idx=outer_train_idx,
            specs=current_specs,
        )
        self._record_inner_rows(baseline_rows, accepted=True)
        accepted_additions = 0

        for iteration in range(1, self.search_config.max_iterations + 1):
            current_names = set(_spec_names(current_specs))
            screened = self._screen_broad_candidates_for_context(
                outer_train_idx=outer_train_idx,
                current_specs=current_specs,
                proposals=proposals,
            )
            available_items = select_screened_candidates(
                [item for item in screened if item["candidate_id"] not in current_names],
                top_k=self.search_config.broad_screen_top_k,
            )
            available_ids = {item["candidate_id"] for item in available_items}
            for item in screened:
                item["kept_for_cv"] = item["candidate_id"] in available_ids

            available_by_name = {
                item["candidate_id"]: item["screened_spec"]
                for item in available_items
                if item.get("screened_spec") is not None
            }
            context = self._build_broad_selection_context(
                outer_fold=outer_fold,
                iteration=iteration,
                train_idx=outer_train_idx,
                current_specs=current_specs,
                current_summary=baseline_summary,
                available_items=available_items,
            )

            try:
                raw_proposals = self.proposal_agent.propose(context)
            except Exception as exc:
                error_payload = {
                    "error": repr(exc),
                    "context": context,
                }
                if self.search_config.save_agent_raw_output:
                    error_payload["agent_raw_output"] = _get_agent_response_trace(
                        self.proposal_agent
                    )
                self._record_decision(
                    outer_fold,
                    iteration,
                    "agent_proposal_error",
                    error_payload,
                )
                self._save_decision_events()
                raise

            effective_raw_proposals = _enrich_broad_selection_proposals(
                raw_proposals,
                available_by_name,
            )
            proposal_payload = {
                "raw_count": len(raw_proposals),
                "valid_count": None,
                "rejected": None,
                "context": context,
                "raw_proposals": raw_proposals,
                "effective_proposals": effective_raw_proposals,
            }
            if self.search_config.save_agent_raw_output:
                proposal_payload["agent_raw_output"] = _get_agent_response_trace(
                    self.proposal_agent
                )
            selected_proposals, rejected = validate_agentic_proposals(
                effective_raw_proposals,
                current_specs=current_specs,
                search_config=self.search_config,
                allow_removals=accepted_additions > 0,
                max_additions=self.search_config.max_additions_per_iter,
            )
            proposal_payload["valid_count"] = len(selected_proposals)
            proposal_payload["rejected"] = rejected
            self._record_decision(
                outer_fold,
                iteration,
                "agent_proposals",
                proposal_payload,
            )

            if not selected_proposals:
                self._record_broad_screening(outer_fold, iteration, screened)
                logger.info(
                    "Outer fold %s broad_screen iteration %s: no valid proposals",
                    outer_fold,
                    iteration,
                )
                break

            candidate_results = []
            screened_by_name = {item["candidate_id"]: item for item in screened}
            for candidate_id, proposal_group in _candidate_groups(selected_proposals):
                candidate_specs = apply_proposals(current_specs, proposal_group)
                if _spec_signature(candidate_specs) == _spec_signature(current_specs):
                    continue
                self.dataset = self.extraction_provider.ensure_features(
                    self.dataset,
                    candidate_specs,
                )
                proposal_specs = _candidate_proposal_specs(
                    current_specs=current_specs,
                    candidate_specs=candidate_specs,
                    proposal_group=proposal_group,
                )
                role_diagnostics = evaluate_candidate_role_diagnostics(
                    dataset=self.dataset.iloc[outer_train_idx],
                    current_specs=current_specs,
                    candidate_specs=proposal_specs,
                    config=self.config,
                    search_config=self.search_config,
                )
                coverage_failures = _coverage_failures(
                    self.dataset.iloc[outer_train_idx],
                    proposal_specs,
                    self.search_config.min_feature_coverage,
                )
                if coverage_failures:
                    summary = {"coverage_failures": coverage_failures}
                    if role_diagnostics:
                        summary["role_diagnostics"] = role_diagnostics
                    candidate_results.append(
                        {
                            "candidate_id": candidate_id,
                            "proposal_group": proposal_group,
                            "specs": candidate_specs,
                            "rows": [],
                            "summary": summary,
                            "comparison": {
                                "passes_acceptance": False,
                                "rejection_reason": "low_feature_coverage",
                                "coverage_failures": coverage_failures,
                            },
                        }
                    )
                    continue

                rows, summary = self._evaluate_inner_cv(
                    outer_fold=outer_fold,
                    iteration=iteration,
                    candidate_name=candidate_id,
                    train_idx=outer_train_idx,
                    specs=candidate_specs,
                )
                if role_diagnostics:
                    summary = dict(summary)
                    summary["role_diagnostics"] = role_diagnostics
                comparison = compare_candidate_to_baseline(
                    baseline_rows=baseline_rows,
                    candidate_rows=rows,
                    search_config=self.search_config,
                )
                candidate_results.append(
                    {
                        "candidate_id": candidate_id,
                        "proposal_group": proposal_group,
                        "specs": candidate_specs,
                        "rows": rows,
                        "summary": summary,
                        "comparison": comparison,
                    }
                )
                self._record_inner_rows(rows, accepted=False)

            accepted = _choose_accepted_candidate(candidate_results)
            candidate_payloads = []
            for item in candidate_results:
                accepted_item = accepted is item
                for proposal in item["proposal_group"]:
                    if proposal.action != "add":
                        continue
                    screened_item = screened_by_name.get(proposal.name)
                    if screened_item is None:
                        continue
                    if item["candidate_id"] != proposal.name and not accepted_item:
                        continue
                    screened_item["cv_comparison"] = item["comparison"]
                    screened_item["cv_accepted"] = bool(
                        accepted_item or screened_item.get("cv_accepted", False)
                    )
                candidate_payloads.append(
                    {
                        "candidate_id": item["candidate_id"],
                        "proposals": [asdict(p) for p in item["proposal_group"]],
                        "summary": item["summary"],
                        "comparison": item["comparison"],
                        "accepted": accepted_item,
                    }
                )

            self._record_broad_screening(outer_fold, iteration, screened)
            self._record_decision(
                outer_fold,
                iteration,
                "candidate_evaluations",
                candidate_payloads,
            )

            if accepted is None:
                logger.info(
                    "Outer fold %s broad_screen iteration %s: no candidate passed acceptance thresholds",
                    outer_fold,
                    iteration,
                )
                if self.search_config.stop_after_rejected_iteration:
                    break
                continue

            current_specs = accepted["specs"]
            baseline_rows = accepted["rows"]
            baseline_summary = accepted["summary"]
            accepted_additions += sum(
                1 for proposal in accepted["proposal_group"] if proposal.action == "add"
            )
            self._record_inner_rows(accepted["rows"], accepted=True)
            self.feature_set_rows.append(
                {
                    "outer_fold": outer_fold,
                    "iteration": iteration,
                    "stage": "accepted_inner",
                    "candidate_id": accepted["candidate_id"],
                    "features": [_spec_to_dict(spec) for spec in current_specs],
                }
            )
            logger.info(
                "Outer fold %s broad_screen iteration %s: accepted %s",
                outer_fold,
                iteration,
                accepted["candidate_id"],
            )

        return current_specs

    def _screen_broad_candidates_for_context(
        self,
        outer_train_idx: np.ndarray,
        current_specs: List[ExplicitFeatureSpec],
        proposals: Sequence[AgenticFeatureProposal],
    ) -> List[Dict[str, Any]]:
        if not proposals:
            return []
        return screen_agentic_candidate_specs(
            dataset=self.dataset.iloc[outer_train_idx],
            current_specs=current_specs,
            proposals=proposals,
            config=self.config,
            search_config=self.search_config,
        )

    def _record_broad_screening(
        self,
        outer_fold: int,
        iteration: int,
        screened: Sequence[Dict[str, Any]],
    ) -> None:
        for item in screened:
            self.screening_metric_rows.append(
                _screening_metric_row(
                    outer_fold=outer_fold,
                    iteration=iteration,
                    item=item,
                )
            )
        self._record_decision(
            outer_fold,
            iteration,
            "broad_screening",
            [
                _screening_decision_payload(item)
                for item in sorted(screened, key=lambda row: row["rank"])
            ],
        )

    def _resolve_selected_aliases(
        self,
        outer_fold: int,
        selected_specs: List[ExplicitFeatureSpec],
    ) -> List[ExplicitFeatureSpec]:
        if not selected_specs:
            return selected_specs
        if not _proposal_agent_supports_alias_resolution(self.proposal_agent):
            return selected_specs

        initial_names = {
            _normalize_feature_name(spec.name)
            for spec in self.initial_specs
            if _normalize_feature_name(spec.name)
        }
        known_specs = [
            spec for spec in selected_specs if _normalize_feature_name(spec.name) in initial_names
        ]
        add_specs = [
            spec
            for spec in selected_specs
            if _normalize_feature_name(spec.name) not in initial_names
        ]
        add_proposals = [_proposal_from_spec(spec) for spec in add_specs]
        if not add_proposals:
            return selected_specs
        if len(add_proposals) < 2 and not known_specs:
            return selected_specs

        context = {
            "prompt_version": "multi_model_agentic_alias_resolution_v1",
            "agentic_path": "agentic_explicit_feature_forest",
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
                "Agentic explicit alias resolution failed; using selected names",
                exc_info=True,
            )
            self._record_decision(
                outer_fold,
                0,
                "alias_resolution",
                {"error": str(exc), "applied_aliases": []},
            )
            return selected_specs

        resolved, applied_aliases = apply_agentic_alias_resolution(
            proposals=add_proposals,
            known_specs=known_specs,
            response=response,
        )
        result: Dict[str, Any] = {
            "context": context,
            "response": response,
            "applied_aliases": applied_aliases,
        }
        if self.search_config.save_agent_raw_output:
            result["agent_raw_output"] = alias_trace
        self._record_decision(outer_fold, 0, "alias_resolution", result)

        if not applied_aliases:
            return selected_specs
        resolved_specs = [
            _proposal_to_spec(proposal) for proposal in resolved if proposal.action == "add"
        ]
        return _dedupe_agentic_specs([*known_specs, *resolved_specs])

    def _harmonize_value_contracts(
        self,
        outer_fold: int,
        selected_specs: List[ExplicitFeatureSpec],
    ) -> List[ExplicitFeatureSpec]:
        if not selected_specs:
            return selected_specs
        if not _proposal_agent_supports_value_harmonization(self.proposal_agent):
            return selected_specs

        context = {
            "prompt_version": "multi_model_agentic_value_harmonization_v1",
            "agentic_path": "agentic_explicit_feature_forest",
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
                "Agentic explicit value harmonization failed; using unharmonized specs",
                exc_info=True,
            )
            self._record_decision(
                outer_fold,
                0,
                "value_harmonization",
                {"error": str(exc), "applied": []},
            )
            return selected_specs

        harmonized, applied = apply_agentic_value_harmonization(
            specs=selected_specs,
            response=response,
        )
        result: Dict[str, Any] = {
            "context": context,
            "response": response,
            "applied": applied,
        }
        if self.search_config.save_agent_raw_output:
            result["agent_raw_output"] = harmonization_trace
        self._record_decision(outer_fold, 0, "value_harmonization", result)
        return harmonized

    def _evaluate_inner_cv(
        self,
        outer_fold: int,
        iteration: int,
        candidate_name: str,
        train_idx: np.ndarray,
        specs: List[ExplicitFeatureSpec],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Evaluate a feature set with inner CV over the outer-training rows."""
        train_df = self.dataset.iloc[train_idx].reset_index(drop=False)
        splits = _make_splits(
            train_df,
            self.config,
            n_splits=self.search_config.inner_folds,
            random_state=self.search_config.random_state + 1000 * outer_fold + iteration,
        )

        rows = []
        for inner_fold, (inner_train_pos, inner_val_pos) in enumerate(splits, start=1):
            inner_train = train_df.iloc[inner_train_pos].set_index("index", drop=True)
            inner_val = train_df.iloc[inner_val_pos].set_index("index", drop=True)
            split_eval = self.evaluator.evaluate_split(
                train_df=inner_train,
                test_df=inner_val,
                specs=specs,
                fold_id=inner_fold,
            )
            rows.append(
                {
                    "outer_fold": outer_fold,
                    "iteration": iteration,
                    "candidate_name": candidate_name,
                    "inner_fold": inner_fold,
                    "feature_names": ",".join(spec.name for spec in specs),
                    **_without_list_values(split_eval.metrics),
                }
            )

        return rows, aggregate_metric_rows(rows)

    def _build_agent_context(
        self,
        outer_fold: int,
        iteration: int,
        train_idx: np.ndarray,
        current_specs: List[ExplicitFeatureSpec],
        current_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build the train-only summary sent to the proposal agent."""
        train_only_df = self.dataset.iloc[train_idx]
        recent_decisions = [
            event for event in self.decision_events if event.get("outer_fold") == outer_fold
        ][-8:]
        return {
            "outer_fold": outer_fold,
            "iteration": iteration,
            "prompt_version": AGENT_PROMPT_VERSION,
            "clinical_question": _clinical_question_text(self.config),
            "estimand": {
                "treatment_column": self.config.treatment_column,
                "outcome_column": self.config.outcome_column,
                "outcome_type": self.config.outcome_type,
            },
            "current_features": [_spec_to_dict(spec) for spec in current_specs],
            "current_inner_cv_metrics": _non_oracle_metrics(current_summary),
            "extraction_summary": summarize_extractions(train_only_df, current_specs),
            "clinical_text_examples": _clinical_text_examples(
                train_only_df,
                self.config.text_column,
                n_examples=self.search_config.clinical_text_examples_per_prompt,
                max_chars=self.search_config.clinical_text_example_chars,
            ),
            "iteration_feedback": build_iteration_feedback(
                recent_decisions,
                self.search_config,
            ),
            "recent_decisions": recent_decisions,
        }

    def _build_broad_inventory_context(
        self,
        outer_fold: int,
        train_idx: np.ndarray,
    ) -> Dict[str, Any]:
        """Build the train-only inventory context before broad extraction."""
        train_only_df = self.dataset.iloc[train_idx]
        required_features = [_spec_to_dict(spec) for spec in self.initial_specs]
        return {
            "outer_fold": outer_fold,
            "iteration": 0,
            "search_mode": "broad_screen",
            "broad_screen_stage": "inventory",
            "prompt_version": BROAD_AGENT_PROMPT_VERSION,
            "broad_candidate_count": self.search_config.broad_candidate_count,
            "broad_screen_top_k": self.search_config.broad_screen_top_k,
            "clinical_question": _clinical_question_text(self.config),
            "estimand": {
                "treatment_column": self.config.treatment_column,
                "outcome_column": self.config.outcome_column,
                "outcome_type": self.config.outcome_type,
            },
            "required_features": required_features,
            "current_features": required_features,
            "clinical_text_examples": _clinical_text_examples(
                train_only_df,
                self.config.text_column,
                n_examples=self.search_config.clinical_text_examples_per_prompt,
                max_chars=self.search_config.clinical_text_example_chars,
            ),
        }

    def _build_broad_selection_context(
        self,
        outer_fold: int,
        iteration: int,
        train_idx: np.ndarray,
        current_specs: List[ExplicitFeatureSpec],
        current_summary: Dict[str, Any],
        available_items: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build the post-extraction broad-screen adaptive selection context."""
        train_only_df = self.dataset.iloc[train_idx]
        recent_decisions = [
            event for event in self.decision_events if event.get("outer_fold") == outer_fold
        ][-8:]
        return {
            "outer_fold": outer_fold,
            "iteration": iteration,
            "search_mode": "broad_screen",
            "broad_screen_stage": "selection",
            "prompt_version": BROAD_AGENT_PROMPT_VERSION,
            "broad_candidate_count": self.search_config.broad_candidate_count,
            "broad_screen_top_k": self.search_config.broad_screen_top_k,
            "clinical_question": _clinical_question_text(self.config),
            "estimand": {
                "treatment_column": self.config.treatment_column,
                "outcome_column": self.config.outcome_column,
                "outcome_type": self.config.outcome_type,
            },
            "required_features": [_spec_to_dict(spec) for spec in self.initial_specs],
            "current_features": [_spec_to_dict(spec) for spec in current_specs],
            "current_inner_cv_metrics": _non_oracle_metrics(current_summary),
            "extraction_summary": summarize_extractions(train_only_df, current_specs),
            "available_extracted_features": _available_extracted_feature_payloads(
                train_only_df,
                available_items,
            ),
            "broad_screen_instructions": [
                "To select an already-extracted feature, return an add proposal with its name from available_extracted_features.",
                "To escape the extracted shortlist, return a new add proposal with a complete extraction contract.",
                "Inner CV acceptance is required before any selected or newly proposed feature becomes part of the current feature set.",
            ],
            "clinical_text_examples": _clinical_text_examples(
                train_only_df,
                self.config.text_column,
                n_examples=self.search_config.clinical_text_examples_per_prompt,
                max_chars=self.search_config.clinical_text_example_chars,
            ),
            "iteration_feedback": build_iteration_feedback(
                recent_decisions,
                self.search_config,
            ),
            "recent_decisions": recent_decisions,
        }

    def _record_inner_rows(self, rows: List[Dict[str, Any]], accepted: bool) -> None:
        for row in rows:
            copied = dict(row)
            copied["accepted_feature_set"] = bool(accepted)
            self.inner_metric_rows.append(copied)

    def _record_decision(
        self,
        outer_fold: int,
        iteration: int,
        event: str,
        payload: Any,
    ) -> None:
        payload = _scrub_decision_payload(
            payload,
            save_agent_context=self.search_config.save_agent_context,
        )
        self.decision_events.append(
            {
                "outer_fold": outer_fold,
                "iteration": iteration,
                "event": event,
                "payload": payload,
            }
        )

    def _save_predictions(self, results_df: pd.DataFrame) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_parquet(self.output_path, index=False)
        logger.info("Agentic predictions saved to: %s", self.output_path)

    def _save_artifacts(self) -> None:
        pd.DataFrame(self.inner_metric_rows).to_csv(
            self.artifact_dir / "inner_cv_metrics.csv",
            index=False,
        )
        pd.DataFrame(self.outer_metric_rows).to_csv(
            self.artifact_dir / "outer_cv_metrics.csv",
            index=False,
        )
        pd.DataFrame(self.screening_metric_rows).to_csv(
            self.artifact_dir / "screening_metrics.csv",
            index=False,
        )
        with open(self.artifact_dir / "feature_sets.json", "w") as f:
            json.dump(self.feature_set_rows, f, indent=2, default=_json_default)
        self._save_decision_events()
        logger.info("Agentic search artifacts saved to: %s", self.artifact_dir)

    def _save_decision_events(self) -> None:
        with open(self.artifact_dir / "agent_decisions.jsonl", "w") as f:
            for event in self.decision_events:
                f.write(json.dumps(event, default=_json_default) + "\n")


class OpenAICompatibleFeatureSearchAgent:
    """LLM proposal agent using an OpenAI-compatible chat completion endpoint."""

    def __init__(self, search_config: AgenticFeatureSearchConfig):
        self.search_config = search_config
        self._client = None
        self._client_pool: Optional[OpenAIClientPool] = None
        self._resolved_agent_model_name: Optional[str] = None
        self._resolved_agent_models_by_url: Dict[str, str] = {}
        self.last_raw_response: Optional[str] = None
        self.last_response_trace: Optional[Dict[str, Any]] = None

    def _ensure_client(self):
        if self._client is not None or self._client_pool is not None:
            return
        self._client_pool = OpenAIClientPool(
            server_urls=self.search_config.agent_server_url,
            api_key=self.search_config.agent_api_key,
            timeout=getattr(self.search_config, "agent_request_timeout", 900.0),
            max_retries=0,
        )

    def _resolve_agent_model_inventory(self) -> Dict[str, str]:
        configured = self.search_config.agent_model_name
        self._ensure_client()
        if self._client is not None:
            server_url = str(self.search_config.agent_server_url or "configured_direct_client")
            if server_url not in self._resolved_agent_models_by_url:
                model_name = (
                    _discover_openai_compatible_model_name(
                        self._client,
                        server_url=server_url,
                        purpose="agent proposal",
                    )
                    if _should_autodiscover_model_name(configured)
                    else str(configured)
                )
                self._resolved_agent_models_by_url[server_url] = model_name
            return dict(self._resolved_agent_models_by_url)

        assert self._client_pool is not None
        urls = list(self._client_pool.server_urls)
        if set(self._resolved_agent_models_by_url) != set(urls):
            resolved: Dict[str, str] = {}
            for server_url in urls:
                if _should_autodiscover_model_name(configured):
                    client = self._client_pool.client_for_url(server_url)
                    resolved[server_url] = _discover_openai_compatible_model_name(
                        client,
                        server_url=server_url,
                        purpose="agent proposal",
                    )
                else:
                    resolved[server_url] = str(configured)
            self._resolved_agent_models_by_url = resolved
        return dict(self._resolved_agent_models_by_url)

    def _resolve_agent_model_name(self) -> str:
        if self._resolved_agent_model_name is None:
            self._resolved_agent_model_name = _endpoint_model_inventory_identity(
                self._resolve_agent_model_inventory()
            )
        return self._resolved_agent_model_name

    def _agent_model_for_url(self, server_url: str) -> str:
        inventory = self._resolve_agent_model_inventory()
        if server_url in inventory:
            return inventory[server_url]
        if len(inventory) == 1:
            return next(iter(inventory.values()))
        raise RuntimeError(f"No resolved agent model identity for endpoint {server_url!r}")

    def _create_completion(self, **kwargs: Any) -> Any:
        self._ensure_client()
        max_attempts = 1 + max(
            0,
            int(getattr(self.search_config, "agent_request_max_retries", 3)),
        )
        retry_kwargs = {
            "max_attempts": max_attempts,
            "initial_delay": float(getattr(self.search_config, "agent_retry_initial_delay", 1.0)),
            "max_delay": float(getattr(self.search_config, "agent_retry_max_delay", 30.0)),
            "backoff_factor": float(getattr(self.search_config, "agent_retry_backoff_factor", 2.0)),
            "context": "agent proposal LLM request",
        }
        if self._client is not None:
            return call_with_exponential_backoff(
                lambda _attempt: self._client.chat.completions.create(**kwargs),
                **retry_kwargs,
            )
        assert self._client_pool is not None
        start_index = self._client_pool.reserve_start_index()

        def operation(attempt: int) -> Any:
            server_url, client = self._client_pool.client_for_attempt(start_index, attempt)
            request_kwargs = dict(kwargs)
            request_kwargs["model"] = self._agent_model_for_url(server_url)
            prompt_chars = sum(
                len(str(message.get("content", "")))
                for message in request_kwargs.get("messages", [])
                if isinstance(message, dict)
            )
            logger.info(
                "Sending agent proposal request to %s model=%s prompt_chars=%.1fK "
                "max_tokens=%s timeout=%s attempt=%s/%s",
                server_url,
                request_kwargs.get("model"),
                prompt_chars / 1000.0,
                request_kwargs.get("max_tokens"),
                getattr(self.search_config, "agent_request_timeout", 900.0),
                attempt + 1,
                max_attempts,
            )
            request_kwargs.update(
                google_json_response_format_kwargs(
                    api_key=self.search_config.agent_api_key,
                    server_url=server_url,
                    model_name=request_kwargs["model"],
                )
            )
            return client.chat.completions.create(**request_kwargs)

        return call_with_exponential_backoff(operation, **retry_kwargs)

    def propose(self, context: Dict[str, Any]) -> Any:
        self._ensure_client()
        self.last_raw_response = None
        self.last_response_trace = None
        prompt = build_agent_prompt(context, self.search_config)
        messages = [{"role": "user", "content": prompt}]
        model_name = self._resolve_agent_model_name()
        attempts: List[Dict[str, Any]] = []
        max_repair_attempts = max(
            0,
            int(getattr(self.search_config, "agent_schema_repair_attempts", 1)),
        )
        is_consensus_disambiguation = context.get("prompt_version") in {
            "agentic_attention_consensus_disambiguation_v1",
            "multi_model_agentic_alias_resolution_v1",
        }
        is_value_harmonization = (
            context.get("prompt_version") == "multi_model_agentic_value_harmonization_v1"
        )
        is_concept_inventory = context.get("prompt_version") in _CONCEPT_INVENTORY_PROMPT_VERSIONS
        is_parsimony_factor = context.get("prompt_version") == _PARSIMONY_FACTOR_PROMPT_VERSION
        is_tfidf_topic_label = context.get("prompt_version") in {
            "tfidf_topic_label_v2",
            "tfidf_topic_recovery_v2",
            "tfidf_orphan_ngram_label_v1",
        }
        is_tfidf_topic_harmonization = context.get("prompt_version") in {
            "tfidf_topic_name_harmonization_v2",
            "tfidf_topic_global_dedup_v2",
            "tfidf_topic_value_harmonization_v2",
            "tfidf_topic_value_repair_v2",
        }
        is_neural_query_prompt = context.get("prompt_version") in {
            "neural_query_feature_v1",
            "neural_query_registry_v1",
            "neural_query_review_v1",
        }
        from .all_evidence_post_extraction_review import (
            POST_EXTRACTION_REVIEW_PROMPT_VERSION,
        )

        is_post_extraction_review = (
            context.get("prompt_version") == POST_EXTRACTION_REVIEW_PROMPT_VERSION
        )
        from .all_evidence_fusion import FUSION_PROMPT_VERSION

        is_all_evidence_fusion = context.get("prompt_version") == FUSION_PROMPT_VERSION

        for attempt_idx in range(max_repair_attempts + 1):
            response_kwargs = {
                "model": model_name,
                "messages": messages,
                "temperature": self.search_config.agent_temperature,
                "max_tokens": self.search_config.agent_max_tokens,
            }
            if is_all_evidence_fusion or is_post_extraction_review:
                # OpenAI-compatible servers, including current vLLM releases,
                # can constrain generation to one syntactically valid JSON
                # object. Semantic validation and repair still run below.
                response_kwargs["response_format"] = {"type": "json_object"}
            enable_thinking = getattr(self.search_config, "agent_enable_thinking", None)
            if (
                enable_thinking is not None
                and str(getattr(self.search_config, "agent_provider", "openai")).strip().lower()
                == "openai"
            ):
                extra_body = {"chat_template_kwargs": {"enable_thinking": bool(enable_thinking)}}
                thinking_token_budget = getattr(
                    self.search_config,
                    "agent_thinking_token_budget",
                    None,
                )
                if bool(enable_thinking) and thinking_token_budget is not None:
                    extra_body["thinking_token_budget"] = int(thinking_token_budget)
                response_kwargs["extra_body"] = extra_body
            response_kwargs.update(
                google_json_response_format_kwargs(
                    api_key=self.search_config.agent_api_key,
                    server_url=self.search_config.agent_server_url,
                    model_name=model_name,
                )
            )
            response = self._create_completion(
                **response_kwargs,
            )
            choice = response.choices[0]
            message = choice.message
            content = message.content or ""
            self.last_raw_response = content
            trace = _chat_completion_trace(
                response=response,
                choice=choice,
                message=message,
                content=content,
            )
            attempts.append(trace)
            self.last_response_trace = _trace_with_repair_attempts(trace, attempts)

            parsed_json_object = False
            parsed: Any = None
            try:
                if (
                    is_consensus_disambiguation
                    or is_value_harmonization
                    or is_concept_inventory
                    or is_parsimony_factor
                    or is_tfidf_topic_label
                    or is_tfidf_topic_harmonization
                    or is_neural_query_prompt
                    or is_post_extraction_review
                    or is_all_evidence_fusion
                ):
                    parsed = parse_agent_json_object(content)
                    parsed_json_object = True
                    if is_value_harmonization:
                        issues = value_harmonization_response_issues(parsed, context)
                    elif is_concept_inventory:
                        issues = concept_inventory_response_issues(parsed, context)
                    elif is_parsimony_factor:
                        issues = parsimony_factor_response_issues(parsed, context)
                    elif is_tfidf_topic_label:
                        from .tfidf_topic_agentic_forest import topic_label_response_issues

                        issues = topic_label_response_issues(parsed, context)
                    elif is_tfidf_topic_harmonization:
                        from .tfidf_topic_agentic_forest import (
                            topic_harmonization_response_issues,
                        )

                        issues = topic_harmonization_response_issues(parsed, context)
                    elif is_neural_query_prompt:
                        from .neural_query_agentic_forest import (
                            QUERY_FEATURE_PROMPT_VERSION,
                            QUERY_REVIEW_PROMPT_VERSION,
                            query_feature_response_issues,
                            query_registry_response_issues,
                            query_review_response_issues,
                        )

                        neural_prompt_version = context.get("prompt_version")
                        if neural_prompt_version == QUERY_FEATURE_PROMPT_VERSION:
                            issues = query_feature_response_issues(parsed, context)
                        elif neural_prompt_version == QUERY_REVIEW_PROMPT_VERSION:
                            issues = query_review_response_issues(parsed, context)
                        else:
                            issues = query_registry_response_issues(parsed, context)
                    elif is_post_extraction_review:
                        from .all_evidence_post_extraction_review import (
                            _normalize_fresh_post_extraction_review_response,
                            post_extraction_review_response_issues,
                        )

                        parsed, normalization_audit = (
                            _normalize_fresh_post_extraction_review_response(
                                parsed,
                                context,
                            )
                        )
                        trace["fresh_response_normalization"] = normalization_audit
                        attempts[-1] = trace
                        self.last_response_trace = _trace_with_repair_attempts(
                            trace,
                            attempts,
                        )
                        issues = post_extraction_review_response_issues(parsed, context)
                    elif is_all_evidence_fusion:
                        from .all_evidence_fusion import (
                            _normalize_agent_response_citation_families,
                            all_evidence_fusion_response_issues,
                        )

                        parsed, normalization_audit = _normalize_agent_response_citation_families(
                            parsed, context
                        )
                        trace["fresh_response_normalization"] = normalization_audit
                        # Retain the original trace field for consumers that
                        # inspect citation-family correction specifically.
                        trace["citation_family_normalization"] = normalization_audit[
                            "citation_family_normalization"
                        ]
                        attempts[-1] = trace
                        self.last_response_trace = _trace_with_repair_attempts(
                            trace,
                            attempts,
                        )
                        issues = all_evidence_fusion_response_issues(parsed, context)
                    else:
                        issues = consensus_disambiguation_response_issues(parsed)
                    if issues:
                        raise ValueError("; ".join(issues))
                    return parsed
                proposals = parse_agent_response(content)
            except Exception as exc:
                failure_kind = "semantic validation" if parsed_json_object else "malformed JSON"
                issues = [f"{failure_kind}: {exc}"]
                if attempt_idx < max_repair_attempts:
                    logger.warning(
                        "Agent response failed %s on attempt %s/%s: "
                        "finish_reason=%s content_chars=%s max_tokens=%s. "
                        "Asking model to repair JSON.",
                        failure_kind,
                        attempt_idx + 1,
                        max_repair_attempts + 1,
                        getattr(choice, "finish_reason", None),
                        len(content),
                        self.search_config.agent_max_tokens,
                    )
                    if is_value_harmonization:
                        repair_prompt = build_value_harmonization_repair_prompt(issues)
                    elif is_concept_inventory:
                        repair_prompt = build_concept_inventory_repair_prompt(issues)
                    elif is_parsimony_factor:
                        repair_prompt = build_parsimony_factor_repair_prompt(issues)
                    elif is_tfidf_topic_label:
                        repair_prompt = (
                            "Repair the topic response as exactly one JSON object with "
                            "general_topic, topic_quality, and proposals. Every proposal "
                            "must cite one or more exact supplied supporting_terms. Problems: "
                            + "; ".join(issues)
                        )
                    elif is_tfidf_topic_harmonization:
                        repair_prompt = (
                            "Repair the harmonization response as exactly one JSON object "
                            "following the original response contract. Cover every required "
                            "candidate exactly once, use only supplied ids/names, and never "
                            "return a review state. Problems: " + "; ".join(issues)
                        )
                    elif is_neural_query_prompt:
                        repair_prompt = (
                            "Repair the neural-query response as exactly one JSON "
                            "object following the original response contract. Use "
                            "only supplied evidence/candidate ids, obey all count "
                            "limits, and cover required candidates exactly once. "
                            "Problems: " + "; ".join(issues)
                        )
                    elif is_post_extraction_review:
                        from .all_evidence_post_extraction_review import (
                            build_post_extraction_review_repair_prompt,
                        )

                        repair_prompt = build_post_extraction_review_repair_prompt(
                            issues,
                            context=context,
                            failed_response=parsed if parsed_json_object else None,
                        )
                    elif is_all_evidence_fusion:
                        from .all_evidence_fusion import (
                            build_all_evidence_fusion_repair_prompt,
                        )

                        repair_prompt = build_all_evidence_fusion_repair_prompt(
                            issues,
                            context,
                        )
                    elif is_consensus_disambiguation:
                        repair_prompt = build_consensus_disambiguation_repair_prompt(issues)
                    else:
                        repair_prompt = build_agent_repair_prompt(issues)
                    messages.extend(
                        [
                            {"role": "assistant", "content": content},
                            {"role": "user", "content": repair_prompt},
                        ]
                    )
                    continue
                if is_post_extraction_review:
                    from .all_evidence_post_extraction_review import (
                        PostExtractionReviewResponseExhausted,
                    )

                    raise PostExtractionReviewResponseExhausted(
                        "remote post-extraction reviewer exhausted bounded response repair"
                    ) from exc
                raise ValueError(
                    "Agent response could not be parsed after "
                    f"{attempt_idx + 1} attempt(s): {issues[0]}"
                ) from exc

            issues = agent_response_schema_issues(proposals, context=context)
            if not issues:
                return proposals

            if attempt_idx < max_repair_attempts:
                logger.warning(
                    "Agent response had schema issues on attempt %s/%s: "
                    "finish_reason=%s content_chars=%s max_tokens=%s issues=%s. "
                    "Asking model to repair JSON.",
                    attempt_idx + 1,
                    max_repair_attempts + 1,
                    getattr(choice, "finish_reason", None),
                    len(content),
                    self.search_config.agent_max_tokens,
                    "; ".join(issues[:3]),
                )
                messages.extend(
                    [
                        {"role": "assistant", "content": content},
                        {"role": "user", "content": build_agent_repair_prompt(issues)},
                    ]
                )
                continue

            logger.warning(
                "Agent proposal response still has schema issues after %s attempt(s): %s",
                attempt_idx + 1,
                "; ".join(issues),
            )
            return proposals

        return []


@dataclass
class CodexCLIResponse:
    content: str
    command: List[str]
    stdout: str
    stderr: str
    returncode: int


def _codex_cli_provider(value: Any) -> bool:
    normalized = str(value or "openai").strip().lower().replace("-", "_")
    return normalized in {"codex", "codex_cli"}


def _codex_optional_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "profile", "default"}:
        return None
    return text


def _codex_extra_args(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return shlex.split(value)
    args: List[str] = []
    for item in value:
        if item is None:
            continue
        if isinstance(item, str):
            args.append(item)
        else:
            args.append(str(item))
    return args


def _codex_exec_command(
    *,
    executable: str,
    model_name: Any,
    reasoning_effort: Any,
    extra_args: Any,
    output_path: Path,
) -> List[str]:
    cmd = [
        os.path.expanduser(str(executable or "codex")),
        "exec",
        "--skip-git-repo-check",
        "--ephemeral",
        "--ignore-rules",
        "--sandbox",
        "read-only",
        "--color",
        "never",
    ]
    model = _codex_optional_value(model_name)
    if model is not None:
        cmd.extend(["--model", model])
    effort = _codex_optional_value(reasoning_effort)
    if effort is not None:
        cmd.extend(["-c", f"model_reasoning_effort={json.dumps(effort)}"])
    cmd.extend(_codex_extra_args(extra_args))
    cmd.extend(["--output-last-message", str(output_path), "-"])
    return cmd


def _tail_text(text: str, limit: int = 4000) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[-limit:]


def _run_codex_exec(
    prompt: str,
    *,
    executable: str,
    model_name: Any,
    reasoning_effort: Any,
    extra_args: Any,
    timeout: Optional[float],
) -> CodexCLIResponse:
    with tempfile.TemporaryDirectory(prefix="oci_codex_cli_") as tmpdir:
        output_path = Path(tmpdir) / "last_message.txt"
        cmd = _codex_exec_command(
            executable=executable,
            model_name=model_name,
            reasoning_effort=reasoning_effort,
            extra_args=extra_args,
            output_path=output_path,
        )
        completed = subprocess.run(
            cmd,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        content = ""
        if output_path.exists():
            content = output_path.read_text()
        if not content.strip():
            content = completed.stdout
        response = CodexCLIResponse(
            content=content.strip(),
            command=cmd,
            stdout=completed.stdout,
            stderr=completed.stderr,
            returncode=int(completed.returncode),
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "codex exec failed with return code "
                f"{completed.returncode}: {_tail_text(completed.stderr or completed.stdout)}"
            )
        if not response.content:
            raise RuntimeError("codex exec returned an empty final message")
        return response


def _codex_backend_prompt(task_prompt: str, *, task_kind: str) -> str:
    return f"""You are being used as a noninteractive JSON LLM backend for {task_kind}.
Do not inspect files, run shell commands, browse, invoke tools, edit files, or use skills.
Use only the information in this prompt. Think internally, then return only the requested JSON.
Do not include markdown fences, comments, prose, or reasoning in the final answer.

{task_prompt}
"""


def _codex_extraction_prompt(
    clinical_text: str,
    specs: List[ExplicitFeatureSpec],
    *,
    max_text_length: Optional[int] = None,
    context_strategy: str = "tail",
    source_text_temporally_valid_by_design: bool = False,
) -> str:
    task_prompt = build_extraction_prompt(
        clinical_text,
        specs,
        max_text_length=max_text_length,
        context_strategy=context_strategy,
        source_text_temporally_valid_by_design=source_text_temporally_valid_by_design,
    )
    complete_document = (
        str(context_strategy).strip().lower().replace("-", "_") == "tail"
        and max_text_length is None
        and not str(clinical_text).startswith("[oci_colbert_v1]")
    )
    reading_instruction = (
        "Read the whole clinical note in the extraction prompt below from beginning " "to end."
        if complete_document
        else "Read every labeled verbatim excerpt in the extraction prompt below."
    )
    extraction_source = (
        "the full note text" if complete_document else "the provided contract-guided excerpts"
    )
    return _codex_backend_prompt(
        f"{reading_instruction} Do not use regex-only, keyword-only, or hand-coded "
        f"rule shortcuts; perform the clinical extraction directly from "
        f"{extraction_source}. If the "
        "note does not support a value, return null for that field.\n\n"
        f"{task_prompt}",
        task_kind="clinical explicit-feature extraction",
    )


def _codex_response_trace(response: CodexCLIResponse, *, model_name: Any) -> Dict[str, Any]:
    return {
        "provider": "codex_cli",
        "model": _codex_optional_value(model_name),
        "raw_content": response.content,
        "returncode": response.returncode,
        "command": response.command,
        "stdout_tail": _tail_text(response.stdout),
        "stderr_tail": _tail_text(response.stderr),
    }


class CodexCLIFeatureSearchAgent:
    """Proposal agent that shells out to `codex exec` for each LLM call."""

    supports_alias_resolution = True
    supports_value_harmonization = True

    def __init__(self, search_config: AgenticFeatureSearchConfig):
        self.search_config = search_config
        self.last_raw_response: Optional[str] = None
        self.last_response_trace: Optional[Dict[str, Any]] = None

    def _run(self, prompt: str) -> CodexCLIResponse:
        max_attempts = 1 + max(
            0,
            int(getattr(self.search_config, "agent_request_max_retries", 3)),
        )
        for attempt in range(max_attempts):
            try:
                return _run_codex_exec(
                    prompt,
                    executable=getattr(self.search_config, "codex_cli_executable", "codex"),
                    model_name=getattr(
                        self.search_config,
                        "codex_cli_model_name",
                        "gpt-5.4-mini",
                    ),
                    reasoning_effort=getattr(
                        self.search_config,
                        "codex_cli_reasoning_effort",
                        "medium",
                    ),
                    extra_args=getattr(self.search_config, "codex_cli_extra_args", []),
                    timeout=getattr(self.search_config, "agent_request_timeout", 900.0),
                )
            except Exception as exc:
                if attempt >= max_attempts - 1:
                    raise
                delay = retry_delay(
                    attempt,
                    initial_delay=float(
                        getattr(self.search_config, "agent_retry_initial_delay", 1.0)
                    ),
                    max_delay=float(getattr(self.search_config, "agent_retry_max_delay", 30.0)),
                    backoff_factor=float(
                        getattr(self.search_config, "agent_retry_backoff_factor", 2.0)
                    ),
                )
                logger.warning(
                    "Codex CLI agent request failed on attempt %s/%s with %s: %s. "
                    "Retrying in %.2fs.",
                    attempt + 1,
                    max_attempts,
                    exc.__class__.__name__,
                    exc,
                    delay,
                )
                import time

                time.sleep(delay)
        raise RuntimeError("Codex CLI agent request failed without an exception")

    def propose(self, context: Dict[str, Any]) -> Any:
        self.last_raw_response = None
        self.last_response_trace = None
        prompt_version = context.get("prompt_version")
        task_kind = (
            "clinical text concept inventory"
            if prompt_version in _CONCEPT_INVENTORY_PROMPT_VERSIONS
            else (
                "clinical latent-factor parsimony review"
                if prompt_version == _PARSIMONY_FACTOR_PROMPT_VERSION
                else "agentic causal-feature proposal"
            )
        )
        base_prompt = _codex_backend_prompt(
            build_agent_prompt(context, self.search_config),
            task_kind=task_kind,
        )
        prompt = base_prompt
        attempts: List[Dict[str, Any]] = []
        max_repair_attempts = max(
            0,
            int(getattr(self.search_config, "agent_schema_repair_attempts", 1)),
        )
        is_consensus_disambiguation = prompt_version in {
            "agentic_attention_consensus_disambiguation_v1",
            "multi_model_agentic_alias_resolution_v1",
        }
        is_value_harmonization = prompt_version == "multi_model_agentic_value_harmonization_v1"
        is_concept_inventory = prompt_version in _CONCEPT_INVENTORY_PROMPT_VERSIONS
        is_parsimony_factor = prompt_version == _PARSIMONY_FACTOR_PROMPT_VERSION
        is_tfidf_topic_label = prompt_version in {
            "tfidf_topic_label_v2",
            "tfidf_topic_recovery_v2",
            "tfidf_orphan_ngram_label_v1",
        }
        is_tfidf_topic_harmonization = prompt_version in {
            "tfidf_topic_name_harmonization_v2",
            "tfidf_topic_global_dedup_v2",
            "tfidf_topic_value_harmonization_v2",
            "tfidf_topic_value_repair_v2",
        }
        is_neural_query_prompt = prompt_version in {
            "neural_query_feature_v1",
            "neural_query_registry_v1",
            "neural_query_review_v1",
        }
        from .all_evidence_post_extraction_review import (
            POST_EXTRACTION_REVIEW_PROMPT_VERSION,
        )

        is_post_extraction_review = prompt_version == POST_EXTRACTION_REVIEW_PROMPT_VERSION

        for attempt_idx in range(max_repair_attempts + 1):
            response = self._run(prompt)
            content = response.content
            self.last_raw_response = content
            trace = _codex_response_trace(
                response,
                model_name=getattr(self.search_config, "codex_cli_model_name", None),
            )
            attempts.append(trace)
            self.last_response_trace = _trace_with_repair_attempts(trace, attempts)

            parsed_json_object = False
            parsed: Any = None
            try:
                if (
                    is_consensus_disambiguation
                    or is_value_harmonization
                    or is_concept_inventory
                    or is_parsimony_factor
                    or is_tfidf_topic_label
                    or is_tfidf_topic_harmonization
                    or is_neural_query_prompt
                    or is_post_extraction_review
                ):
                    parsed = parse_agent_json_object(content)
                    parsed_json_object = True
                    if is_value_harmonization:
                        issues = value_harmonization_response_issues(parsed, context)
                    elif is_concept_inventory:
                        issues = concept_inventory_response_issues(parsed, context)
                    elif is_parsimony_factor:
                        issues = parsimony_factor_response_issues(parsed, context)
                    elif is_tfidf_topic_label:
                        from .tfidf_topic_agentic_forest import topic_label_response_issues

                        issues = topic_label_response_issues(parsed, context)
                    elif is_tfidf_topic_harmonization:
                        from .tfidf_topic_agentic_forest import (
                            topic_harmonization_response_issues,
                        )

                        issues = topic_harmonization_response_issues(parsed, context)
                    elif is_neural_query_prompt:
                        from .neural_query_agentic_forest import (
                            QUERY_FEATURE_PROMPT_VERSION,
                            QUERY_REVIEW_PROMPT_VERSION,
                            query_feature_response_issues,
                            query_registry_response_issues,
                            query_review_response_issues,
                        )

                        if prompt_version == QUERY_FEATURE_PROMPT_VERSION:
                            issues = query_feature_response_issues(parsed, context)
                        elif prompt_version == QUERY_REVIEW_PROMPT_VERSION:
                            issues = query_review_response_issues(parsed, context)
                        else:
                            issues = query_registry_response_issues(parsed, context)
                    elif is_post_extraction_review:
                        from .all_evidence_post_extraction_review import (
                            _normalize_fresh_post_extraction_review_response,
                            post_extraction_review_response_issues,
                        )

                        parsed, normalization_audit = (
                            _normalize_fresh_post_extraction_review_response(
                                parsed,
                                context,
                            )
                        )
                        trace["fresh_response_normalization"] = normalization_audit
                        attempts[-1] = trace
                        self.last_response_trace = _trace_with_repair_attempts(
                            trace,
                            attempts,
                        )
                        issues = post_extraction_review_response_issues(parsed, context)
                    else:
                        issues = consensus_disambiguation_response_issues(parsed)
                    if issues:
                        raise ValueError("; ".join(issues))
                    return parsed
                proposals = parse_agent_response(content)
            except Exception as exc:
                failure_kind = "semantic validation" if parsed_json_object else "malformed JSON"
                issues = [f"{failure_kind}: {exc}"]
                if attempt_idx < max_repair_attempts:
                    logger.warning(
                        "Codex CLI agent response had malformed JSON on attempt %s/%s; "
                        "requesting repaired JSON.",
                        attempt_idx + 1,
                        max_repair_attempts + 1,
                    )
                    if is_value_harmonization:
                        repair_prompt = build_value_harmonization_repair_prompt(issues)
                    elif is_concept_inventory:
                        repair_prompt = build_concept_inventory_repair_prompt(issues)
                    elif is_parsimony_factor:
                        repair_prompt = build_parsimony_factor_repair_prompt(issues)
                    elif is_tfidf_topic_label:
                        repair_prompt = (
                            "Repair the topic response as exactly one JSON object with "
                            "general_topic, topic_quality, and proposals. Every proposal "
                            "must cite exact supplied supporting_terms. Problems: "
                            + "; ".join(issues)
                        )
                    elif is_tfidf_topic_harmonization:
                        repair_prompt = (
                            "Repair the harmonization response as exactly one JSON object "
                            "following the original response contract. Cover every required "
                            "candidate exactly once, use only supplied ids/names, and never "
                            "return a review state. Problems: " + "; ".join(issues)
                        )
                    elif is_neural_query_prompt:
                        repair_prompt = (
                            "Repair the neural-query response as exactly one JSON "
                            "object following the original response contract. Use "
                            "only supplied evidence/candidate ids, obey all count "
                            "limits, and cover required candidates exactly once. "
                            "Problems: " + "; ".join(issues)
                        )
                    elif is_post_extraction_review:
                        from .all_evidence_post_extraction_review import (
                            build_post_extraction_review_repair_prompt,
                        )

                        repair_prompt = build_post_extraction_review_repair_prompt(
                            issues,
                            context=context,
                            failed_response=parsed if parsed_json_object else None,
                        )
                    elif is_consensus_disambiguation:
                        repair_prompt = build_consensus_disambiguation_repair_prompt(issues)
                    else:
                        repair_prompt = build_agent_repair_prompt(issues)
                    prompt = (
                        f"{base_prompt}\n\nPrevious Codex response:\n{content}\n\n"
                        f"{repair_prompt}\nReturn repaired JSON only."
                    )
                    continue
                if is_post_extraction_review:
                    from .all_evidence_post_extraction_review import (
                        PostExtractionReviewResponseExhausted,
                    )

                    raise PostExtractionReviewResponseExhausted(
                        "Codex post-extraction reviewer exhausted bounded response repair"
                    ) from exc
                raise ValueError(
                    "Codex CLI agent response could not be parsed after "
                    f"{attempt_idx + 1} attempt(s): {issues[0]}"
                ) from exc

            issues = agent_response_schema_issues(proposals, context=context)
            if not issues:
                return proposals

            if attempt_idx < max_repair_attempts:
                logger.warning(
                    "Codex CLI agent response had schema issues on attempt %s/%s: %s. "
                    "Requesting repaired JSON.",
                    attempt_idx + 1,
                    max_repair_attempts + 1,
                    "; ".join(issues[:3]),
                )
                prompt = (
                    f"{base_prompt}\n\nPrevious Codex response:\n{content}\n\n"
                    f"{build_agent_repair_prompt(issues)}\nReturn repaired JSON only."
                )
                continue

            logger.warning(
                "Codex CLI proposal response still has schema issues after %s attempt(s): %s",
                attempt_idx + 1,
                "; ".join(issues),
            )
            return proposals

        return []


def make_feature_search_agent(search_config: AgenticFeatureSearchConfig) -> Any:
    if _codex_cli_provider(getattr(search_config, "agent_provider", "openai")):
        return CodexCLIFeatureSearchAgent(search_config)
    return OpenAICompatibleFeatureSearchAgent(search_config)


def _proposal_agent_supports_value_harmonization(agent: Any) -> bool:
    return isinstance(
        agent,
        (OpenAICompatibleFeatureSearchAgent, CodexCLIFeatureSearchAgent),
    ) or bool(getattr(agent, "supports_value_harmonization", False))


def _proposal_agent_supports_alias_resolution(agent: Any) -> bool:
    return isinstance(
        agent,
        (OpenAICompatibleFeatureSearchAgent, CodexCLIFeatureSearchAgent),
    ) or bool(getattr(agent, "supports_alias_resolution", False))


class VLLMExplicitFeatureExtractionProvider:
    """Ensure requested explicit feature columns exist, using grouped extraction."""

    extraction_request_group_dependent = True

    def __init__(self, config: AppliedInferenceConfig, output_dir: Path):
        self.config = config
        self.feature_config = config.explicit_features
        self.output_dir = Path(output_dir)
        self.cache = ExtractionCache(
            cache_dir=self.feature_config.cache_dir or str(self.output_dir)
        )
        self._column_fingerprints: Dict[str, str] = {}
        self._resolved_vllm_model_name: Optional[str] = None
        self._resolved_vllm_models_by_url: Dict[str, str] = {}
        self._active_text_hash: Optional[str] = None
        self._active_patient_text_hashes: List[str] = []
        self._active_request_group_contracts_by_spec: Dict[str, List[Dict[str, Any]]] = {}

    def adaptive_review_contract_local_extraction(self) -> bool:
        """Whether one target has identical request semantics in every registry."""

        return int(getattr(self.feature_config, "max_variables_per_extraction_request", 10)) == 1

    def ensure_features(
        self,
        dataset: pd.DataFrame,
        specs: List[ExplicitFeatureSpec],
    ) -> pd.DataFrame:
        dataset = dataset.copy()
        active_texts = [str(value or "") for value in dataset[self.config.text_column].tolist()]
        self._active_patient_text_hashes = [
            hashlib.sha256(value.encode("utf-8")).hexdigest() for value in active_texts
        ]
        self._active_text_hash = hashlib.sha256(
            json.dumps(
                active_texts,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        # A target's extracted value depends on every companion contract in its
        # request: companions alter both the JSON schema and contract-RAG query.
        # Plan groups before cache lookup so per-spec entries are bound to the
        # exact ordered request group that produced them. Existing unowned
        # columns are intentionally excluded from provider request planning.
        provider_owned_or_missing = []
        for spec in specs:
            value_col = f"explicit_feat_{spec.name}"
            missing_col = f"{value_col}_missing"
            if (
                spec.name in self._column_fingerprints
                or value_col not in dataset.columns
                or missing_col not in dataset.columns
            ):
                provider_owned_or_missing.append(spec)
        planned_groups = self._extraction_spec_groups(provider_owned_or_missing)
        self._active_request_group_contracts_by_spec = {}
        for group in planned_groups:
            contracts = [_spec_extraction_contract_dict(spec) for spec in group]
            for spec in group:
                self._active_request_group_contracts_by_spec[spec.name] = contracts

        missing_specs: List[ExplicitFeatureSpec] = []
        row_caches: Dict[str, pd.DataFrame] = {}
        for spec in provider_owned_or_missing:
            if self._columns_available(dataset, spec):
                continue

            cached = self._load_cached_spec(dataset, spec)
            if cached is not None:
                self._merge_extracted_columns(dataset, cached, [spec])
                continue

            row_cached = self._load_cached_spec_rows(dataset, spec)
            if row_cached is not None:
                complete = self.cache.rows_to_complete_dataframe(
                    row_cached,
                    expected_rows=len(dataset),
                )
                if complete is not None:
                    logger.info(
                        "Using complete row cache for agentic feature extraction: %s",
                        spec.name,
                    )
                    self._merge_extracted_columns(dataset, complete, [spec])
                    self._save_per_spec_caches(complete, [spec])
                    continue
                row_caches[spec.name] = row_cached

            missing_specs.append(spec)

        if missing_specs:
            missing_names = {spec.name for spec in missing_specs}
            for group in planned_groups:
                if not any(spec.name in missing_names for spec in group):
                    continue
                if self.feature_config.cache_enabled:
                    extracted_df = self._extract_spec_group_resumable(
                        dataset,
                        group,
                        row_caches,
                    )
                else:
                    extracted_df = self._extract_spec_group_with_fallback(dataset, group)
                self._merge_extracted_columns(dataset, extracted_df, group)
                self._save_per_spec_caches(extracted_df, group)
        return dataset

    def _extraction_spec_groups(
        self,
        specs: List[ExplicitFeatureSpec],
    ) -> List[List[ExplicitFeatureSpec]]:
        """Group related contracts while enforcing the hard ten-variable cap."""
        if self.feature_config.extraction_context_strategy in {COMPLETE_PAGED_VERSION, "colbert"}:
            # Retrieval selects evidence independently for each question.
            # Complete-page responses carry one closed feature contract so that
            # every page can later be reconciled without companion-variable
            # prompt effects.
            return [[spec] for spec in specs]
        maximum = int(getattr(self.feature_config, "max_variables_per_extraction_request", 10))
        if not 1 <= maximum <= 10:
            raise ValueError("max_variables_per_extraction_request must be in [1, 10]")
        grouping_strategy = (
            str(getattr(self.feature_config, "extraction_grouping_strategy", "clinical_domain"))
            .strip()
            .lower()
            .replace("-", "_")
        )
        if grouping_strategy == "packed":
            groups = [specs[start : start + maximum] for start in range(0, len(specs), maximum)]
            if any(len(group) > 10 for group in groups):  # pragma: no cover
                raise RuntimeError("Extraction grouping exceeded the hard ten-variable cap")
            return groups
        if grouping_strategy != "clinical_domain":
            raise ValueError("extraction_grouping_strategy must be 'clinical_domain' or 'packed'")
        domains: Dict[str, List[ExplicitFeatureSpec]] = {}
        domain_order: List[str] = []

        def broad_domain_family(domain: str, parent_object: str, feature_name: str) -> str:
            """Normalize agent vocabulary into stable prompt-sized domains.

            Name harmonization deliberately retains specific clinical domains in
            each contract.  Agents nevertheless use near-synonyms (for example
            ``radiology``, ``imaging``, and ``neuroimaging``), which otherwise
            creates mostly singleton extraction requests.  These broad families
            affect request packing only; the original domain/parent remains in
            the feature contract and cache identity.
            """
            normalized_domain = re.sub(r"[^a-z0-9]+", "_", domain.lower()).strip("_")
            name = re.sub(r"[^a-z0-9]+", "_", feature_name.lower()).strip("_")
            parent = re.sub(r"[^a-z0-9]+", "_", parent_object.lower()).strip("_")

            if any(
                token in name
                for token in (
                    "treatment",
                    "therapy",
                    "regimen",
                    "radiation_referral",
                    "medication",
                    "surgical_history",
                    "procedure",
                    "resection",
                )
            ):
                return "treatment_procedures_and_care"
            if normalized_domain in {
                "imaging",
                "radiology",
                "neuroimaging",
                "nuclear_medicine",
            } or any(token in name for token in ("imaging", "suvmax")):
                return "imaging_and_staging"
            if normalized_domain in {
                "hematology",
                "laboratory",
                "laboratory_medicine",
                "platelet",
                "renal",
                "hepatic",
                "chemistry",
                "vital_signs",
                "anthropometry",
            } or any(
                token in name
                for token in (
                    "count",
                    "serum_",
                    "creatinine",
                    "hemoglobin",
                    "albumin",
                    "blood_pressure",
                    "heart_rate",
                    "respiratory_rate",
                    "oxygen_saturation",
                )
            ):
                return "laboratory_and_vital_measurements"
            if normalized_domain in {
                "oncology",
                "cancer",
                "genetics",
                "genomics",
                "molecular_oncology",
                "pathology",
                "histopathology",
                "histology",
                "tumor_biology",
            }:
                return "cancer_diagnosis_and_biology"
            if normalized_domain in {
                "respiratory",
                "pulmonary",
                "pulmonology",
                "cardiovascular",
                "gastrointestinal",
                "neurology",
                "musculoskeletal",
                "symptoms",
                "physical_examination",
                "functional_status",
                "performance_status",
                "pain",
            }:
                return "symptoms_examination_and_function"
            if normalized_domain in {
                "surgical_history",
                "oncology_treatment",
                "treatment_administration",
                "pharmacology",
                "medication",
                "medication_use",
                "supportive_care",
                "palliative_care",
                "patient_behavior",
            }:
                return "treatment_procedures_and_care"
            # Demographics, social history, comorbid history, preferences, and
            # otherwise uncommon domains are still a coherent baseline-history
            # request family.  Parent/name remain visible in every instruction.
            return "demographics_history_and_preferences"

        for spec in specs:
            text = str(spec.description or spec.name).strip().lower()
            structured = re.match(
                r"^clinical_domain=([^;]+);\s*parent_object=([^:]+):",
                text,
            )
            if structured:
                # A prompt may contain variables from different reusable parent
                # objects when they share a clinical domain. Packing at this
                # level preserves clinical coherence while avoiding thousands of
                # mostly singleton patient requests; the hard cap still applies.
                source_domain = structured.group(1).strip()
                parent_object = structured.group(2).strip()
                domain = broad_domain_family(source_domain, parent_object, spec.name)
            elif ":" in text:
                domain = text.split(":", 1)[0]
            else:
                # Legacy/prespecified specs do not carry structured grouping
                # metadata. Pack them efficiently rather than inventing a domain
                # from the first token of every feature name.
                domain = "__ungrouped__"
            if domain not in domains:
                domains[domain] = []
                domain_order.append(domain)
            domains[domain].append(spec)
        groups: List[List[ExplicitFeatureSpec]] = []
        for domain in domain_order:
            values = domains[domain]
            groups.extend(
                values[start : start + maximum] for start in range(0, len(values), maximum)
            )
        if any(len(group) > 10 for group in groups):
            raise RuntimeError("Extraction grouping exceeded the hard ten-variable cap")
        return groups

    def _columns_available(self, dataset: pd.DataFrame, spec: ExplicitFeatureSpec) -> bool:
        value_col = f"explicit_feat_{spec.name}"
        missing_col = f"{value_col}_missing"
        if value_col not in dataset.columns or missing_col not in dataset.columns:
            return False

        known_fingerprint = self._column_fingerprints.get(spec.name)
        if known_fingerprint is None:
            return True
        requested_fingerprint = self._spec_fingerprint(spec)
        if known_fingerprint == requested_fingerprint:
            return True

        logger.warning(
            "Re-extracting agentic feature %s because the requested extraction "
            "contract differs from the in-memory columns",
            spec.name,
        )
        return False

    def _load_cached_spec(
        self,
        dataset: pd.DataFrame,
        spec: ExplicitFeatureSpec,
    ) -> Optional[pd.DataFrame]:
        if not self.feature_config.cache_enabled:
            return None
        cached = self.cache.load_if_valid(
            self._dataset_cache_key(),
            self._cache_config([spec]),
            expected_rows=len(dataset),
        )
        if cached is not None:
            logger.info("Using cached agentic feature extraction: %s", spec.name)
        return cached

    def _load_cached_spec_rows(
        self,
        dataset: pd.DataFrame,
        spec: ExplicitFeatureSpec,
    ) -> Optional[pd.DataFrame]:
        if not self.feature_config.cache_enabled:
            return None
        positional = self.cache.load_rows_if_valid(
            self._dataset_cache_key(),
            self._cache_config([spec]),
            expected_rows=len(dataset),
        )
        patient = self.cache.load_patient_values(
            self._dataset_cache_key(),
            self._cache_config([spec]),
            self._active_patient_text_hashes,
        )
        frames = [frame for frame in (patient, positional) if frame is not None]
        if not frames:
            return None
        return (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates(subset=["__oci_cache_row_index"], keep="last")
            .sort_values("__oci_cache_row_index")
            .reset_index(drop=True)
        )

    def _extract_spec_group_with_fallback(
        self,
        dataset: pd.DataFrame,
        specs: List[ExplicitFeatureSpec],
    ) -> pd.DataFrame:
        try:
            return self._extract_spec_group(dataset, specs)
        except Exception:
            if len(specs) == 1:
                raise
            midpoint = max(1, len(specs) // 2)
            logger.warning(
                "Grouped agentic extraction failed for %s features; splitting into "
                "groups of %s and %s",
                len(specs),
                midpoint,
                len(specs) - midpoint,
                exc_info=True,
            )
            left = self._extract_spec_group_with_fallback(dataset, specs[:midpoint])
            right = self._extract_spec_group_with_fallback(dataset, specs[midpoint:])
            return pd.concat([left, right], axis=1)

    def _extract_spec_group(
        self,
        dataset: pd.DataFrame,
        specs: List[ExplicitFeatureSpec],
    ) -> pd.DataFrame:
        logger.info(
            "Extracting %s agentic feature(s) with LLM: %s",
            len(specs),
            [spec.name for spec in specs],
        )
        model_name = self._resolve_vllm_model_name()
        model_inventory = self._resolve_vllm_model_inventory()
        extractor = VLLMFeatureExtractor(
            specs=specs,
            mode=self.feature_config.vllm_mode,
            server_url=self.feature_config.vllm_server_url or "http://localhost:8000/v1",
            model_name=model_name,
            model_names_by_url=model_inventory,
            tensor_parallel_size=self.feature_config.vllm_tensor_parallel_size,
            gpu_memory_utilization=self.feature_config.vllm_gpu_memory_utilization,
            download_dir=self.feature_config.vllm_download_dir,
            max_model_len=self.feature_config.vllm_max_model_len,
            vllm_reasoning_parser=self.feature_config.vllm_reasoning_parser,
            vllm_enable_thinking=getattr(self.feature_config, "vllm_enable_thinking", None),
            api_key=getattr(self.feature_config, "vllm_api_key", "EMPTY"),
            max_retries=self.feature_config.extraction_max_retries,
            retry_initial_delay=getattr(
                self.feature_config,
                "extraction_retry_initial_delay",
                1.0,
            ),
            retry_max_delay=getattr(
                self.feature_config,
                "extraction_retry_max_delay",
                30.0,
            ),
            retry_backoff_factor=getattr(
                self.feature_config,
                "extraction_retry_backoff_factor",
                2.0,
            ),
            request_timeout=getattr(
                self.feature_config,
                "extraction_request_timeout",
                900.0,
            ),
            temperature=self.feature_config.extraction_temperature,
            max_tokens=self.feature_config.extraction_max_tokens,
            max_text_length=self.feature_config.extraction_max_text_length,
            colbert=self.feature_config.colbert,
            context_strategy=getattr(
                self.feature_config,
                "extraction_context_strategy",
                "colbert",
            ),
            source_text_temporally_valid_by_design=bool(
                getattr(
                    self.feature_config,
                    "source_text_temporally_valid_by_design",
                    False,
                )
            ),
        )
        try:
            if self.feature_config.extraction_context_strategy == COMPLETE_PAGED_VERSION:
                return self._extract_complete_paged_spec(
                    dataset=dataset,
                    specs=specs,
                    extractor=extractor,
                )
            return extractor.extract_to_dataframe(
                dataset[self.config.text_column].tolist(),
                batch_size=self.feature_config.extraction_batch_size,
            )
        finally:
            extractor.cleanup()

    def _extract_complete_paged_spec(
        self,
        *,
        dataset: pd.DataFrame,
        specs: List[ExplicitFeatureSpec],
        extractor: VLLMFeatureExtractor,
    ) -> pd.DataFrame:
        """Extract one feature from every character of every nonempty note."""

        if len(specs) != 1:
            raise ValueError("complete_paged_v1 requires one feature contract per request")
        spec = specs[0]
        feature = CompleteFeatureContract(
            name=spec.name,
            value_type=spec.type,
            description=spec.description or spec.name,
            categories=tuple(spec.categories or ()),
            temporal_rule=spec.temporal_rule,
            aggregation_rule=spec.aggregation_rule,
        )
        geometry = CompletePagingGeometry(
            core_chars=int(self.feature_config.complete_page_core_chars),
            context_chars=int(self.feature_config.complete_page_context_chars),
            max_page_chars=int(self.feature_config.complete_page_max_chars),
        )
        texts = [
            "" if pd.isna(value) else str(value)
            for value in dataset[self.config.text_column].tolist()
        ]
        notes = {str(index): text for index, text in enumerate(texts) if text}
        value_column = f"explicit_feat_{spec.name}"
        missing_column = f"{value_column}_missing"
        rows: List[Dict[str, Any]] = [
            {value_column: None, missing_column: True} for _ in texts
        ]
        if not notes:
            return pd.DataFrame(rows, columns=[value_column, missing_column])

        request_plan = plan_complete_paged_requests(
            notes,
            (feature,),
            geometry=geometry,
        )
        page_results: Dict[str, CompletePageResponse] = {}

        def run_page(request: Any) -> Tuple[str, CompletePageResponse]:
            response = extractor.extract_complete_page(
                text=notes[request.patient_id],
                page=request.page,
                feature=feature,
                geometry=geometry,
            )
            return request.request_id, response

        workers = max(
            1,
            min(
                len(request_plan.requests),
                int(self.feature_config.extraction_batch_size),
            ),
        )
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(run_page, request): request.request_id
                for request in request_plan.requests
            }
            for future in as_completed(futures):
                request_id, response = future.result()
                if request_id in page_results:
                    raise RuntimeError("complete-page extraction duplicated a response")
                page_results[request_id] = response
        if set(page_results) != {request.request_id for request in request_plan.requests}:
            raise RuntimeError("complete-page extraction omitted a planned response")

        reconciliation_ledgers: Dict[str, Mapping[str, Any]] = {}
        final_responses: Dict[str, Mapping[str, Any]] = {}
        for patient_id, text in notes.items():
            requests = [
                request
                for request in request_plan.requests
                if request.patient_id == patient_id
            ]

            def reducer(children: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
                return extractor.reconcile_complete_pages(
                    text=text,
                    feature=feature,
                    children=children,
                )

            final, ledger = reconcile_complete_page_responses(
                [(request.request_id, page_results[request.request_id]) for request in requests],
                reducer=reducer,
                fan_in=int(self.feature_config.complete_reconciliation_fan_in),
            )
            value = final.normalized_value
            if final.status == "positive":
                if spec.type == "categorical":
                    value = str(value)
                    if value not in set(spec.categories or ()):
                        raise ValueError(
                            "complete-page extraction returned an undeclared category"
                        )
                elif (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                ):
                    raise ValueError(
                        "complete-page extraction returned an invalid continuous value"
                    )
                rows[int(patient_id)] = {
                    value_column: value,
                    missing_column: False,
                }
            reconciliation_ledgers[patient_id] = ledger
            final_responses[patient_id] = final.as_dict()

        normalized_pages = {
            request_id: response.as_dict()
            for request_id, response in page_results.items()
        }
        ledger_dir = self.output_dir / "complete_paged_ledgers"
        ledger_dir.mkdir(parents=True, exist_ok=True)
        ledger_name = hashlib.sha256(
            json.dumps(
                {
                    "feature": feature.contract_sha256,
                    "notes": [request.note_sha256 for request in request_plan.requests],
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        ledger_path = ledger_dir / f"{ledger_name}.json"
        ledger_path.write_text(
            json.dumps(
                {
                    "request_plan": request_plan.as_dict(),
                    "coverage": build_complete_paged_coverage_ledger(
                        request_plan,
                        normalized_pages,
                    ),
                    "page_responses": normalized_pages,
                    "reconciliations": reconciliation_ledgers,
                    "final_responses": final_responses,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return pd.DataFrame(rows, columns=[value_column, missing_column])

    def _extract_spec_group_resumable(
        self,
        dataset: pd.DataFrame,
        specs: List[ExplicitFeatureSpec],
        row_caches: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Extract missing rows and flush row-level progress after each batch."""
        expected_rows = len(dataset)
        missing_by_spec = {
            spec.name: self._missing_row_indices_for_spec(
                expected_rows,
                row_caches.get(spec.name),
            )
            for spec in specs
        }
        rows_to_extract = sorted(
            {row_idx for missing_rows in missing_by_spec.values() for row_idx in missing_rows}
        )

        if rows_to_extract:
            logger.info(
                "Resumable agentic extraction will process %s/%s row(s) for %s feature(s)",
                len(rows_to_extract),
                expected_rows,
                len(specs),
            )
        batch_size = max(1, int(self.feature_config.extraction_batch_size))
        for start in range(0, len(rows_to_extract), batch_size):
            batch_indices = rows_to_extract[start : start + batch_size]
            batch_df = dataset.iloc[batch_indices]
            logger.info(
                "Extracting resumable agentic batch rows %s-%s (%s rows)",
                batch_indices[0],
                batch_indices[-1],
                len(batch_indices),
            )
            batch_extracted = self._extract_spec_group_with_fallback(batch_df, specs)
            for spec in specs:
                value_col = f"explicit_feat_{spec.name}"
                missing_col = f"{value_col}_missing"
                self.cache.save_rows(
                    self._dataset_cache_key(),
                    self._cache_config([spec]),
                    batch_indices,
                    batch_extracted[[value_col, missing_col]].copy(),
                )
                self.cache.save_patient_values(
                    self._dataset_cache_key(),
                    self._cache_config([spec]),
                    [self._active_patient_text_hashes[index] for index in batch_indices],
                    batch_extracted[[value_col, missing_col]].copy(),
                )

        return self._load_complete_rows_for_specs(dataset, specs)

    def _missing_row_indices_for_spec(
        self,
        expected_rows: int,
        rows_df: Optional[pd.DataFrame],
    ) -> List[int]:
        if rows_df is None:
            return list(range(expected_rows))
        processed = set(rows_df["__oci_cache_row_index"].astype(int).tolist())
        return [idx for idx in range(expected_rows) if idx not in processed]

    def _load_complete_rows_for_specs(
        self,
        dataset: pd.DataFrame,
        specs: List[ExplicitFeatureSpec],
    ) -> pd.DataFrame:
        frames = []
        for spec in specs:
            rows_df = self._load_cached_spec_rows(dataset, spec)
            complete = self.cache.rows_to_complete_dataframe(
                rows_df,
                expected_rows=len(dataset),
            )
            if complete is None:
                raise ValueError(
                    f"Row-level extraction cache for {spec.name!r} is incomplete after extraction"
                )
            frames.append(complete)
        return pd.concat(frames, axis=1)

    def _merge_extracted_columns(
        self,
        dataset: pd.DataFrame,
        extracted_df: pd.DataFrame,
        specs: List[ExplicitFeatureSpec],
    ) -> None:
        for spec in specs:
            value_col = f"explicit_feat_{spec.name}"
            missing_col = f"{value_col}_missing"
            missing = [col for col in [value_col, missing_col] if col not in extracted_df.columns]
            if missing:
                raise ValueError(
                    f"Extraction result for {spec.name!r} missing expected columns: {missing}"
                )
            dataset[value_col] = extracted_df[value_col].values
            dataset[missing_col] = extracted_df[missing_col].values
            self._column_fingerprints[spec.name] = self._spec_fingerprint(spec)

    def _save_per_spec_caches(
        self,
        extracted_df: pd.DataFrame,
        specs: List[ExplicitFeatureSpec],
    ) -> None:
        if not self.feature_config.cache_enabled:
            return
        for spec in specs:
            value_col = f"explicit_feat_{spec.name}"
            missing_col = f"{value_col}_missing"
            self.cache.save(
                self._dataset_cache_key(),
                self._cache_config([spec]),
                extracted_df[[value_col, missing_col]].copy(),
            )

    def _dataset_cache_key(self) -> str:
        return self.config.dataset_path or "in_memory_dataset"

    def _cache_config(self, specs: List[ExplicitFeatureSpec]) -> Dict[str, Any]:
        model_name = self._resolve_vllm_model_name()
        model_inventory = self._resolve_vllm_model_inventory()
        if len(specs) == 1:
            request_group_contracts = self._active_request_group_contracts_by_spec.get(
                specs[0].name,
                [_spec_extraction_contract_dict(specs[0])],
            )
        else:
            request_group_contracts = [_spec_extraction_contract_dict(spec) for spec in specs]
        request_group_sha256 = hashlib.sha256(
            json.dumps(
                request_group_contracts,
                sort_keys=True,
                separators=(",", ":"),
                default=_json_default,
            ).encode("utf-8")
        ).hexdigest()
        cache_config = {
            "features": [_spec_extraction_contract_dict(spec) for spec in specs],
            "prompt_template_version": EXTRACTION_PROMPT_VERSION,
            "vllm_model_name": model_name,
            "vllm_endpoint_model_inventory": model_inventory,
            "vllm_max_model_len": self.feature_config.vllm_max_model_len,
            "vllm_reasoning_parser": resolve_vllm_reasoning_parser(
                self.feature_config.vllm_reasoning_parser,
                next(iter(model_inventory.values())),
            ),
            "vllm_reasoning_parser_inventory": {
                url: resolve_vllm_reasoning_parser(
                    self.feature_config.vllm_reasoning_parser,
                    endpoint_model,
                )
                for url, endpoint_model in model_inventory.items()
            },
            "vllm_enable_thinking": getattr(self.feature_config, "vllm_enable_thinking", None),
            "extraction_temperature": self.feature_config.extraction_temperature,
            "extraction_max_tokens": self.feature_config.extraction_max_tokens,
            "extraction_max_text_length": self.feature_config.extraction_max_text_length,
            "extraction_grouping_strategy": getattr(
                self.feature_config,
                "extraction_grouping_strategy",
                "clinical_domain",
            ),
            "extraction_context_strategy": getattr(
                self.feature_config,
                "extraction_context_strategy",
                "colbert",
            ),
            "extraction_context_compactor_version": CONTRACT_LEXICAL_CONTEXT_VERSION,
            "extraction_grouping_version": EXTRACTION_GROUPING_VERSION,
            "extraction_request_group_sha256": request_group_sha256,
            "patient_text_hash": self._active_text_hash,
            "source_text_temporally_valid_by_design": bool(
                getattr(
                    self.feature_config,
                    "source_text_temporally_valid_by_design",
                    False,
                )
            ),
            "temporal_boundary_enforced": not bool(
                getattr(
                    self.feature_config,
                    "source_text_temporally_valid_by_design",
                    False,
                )
            ),
            "max_variables_per_extraction_request": int(
                getattr(self.feature_config, "max_variables_per_extraction_request", 10)
            ),
            "extraction_batch_size": int(self.feature_config.extraction_batch_size),
        }
        if self.feature_config.extraction_context_strategy == "colbert":
            from ..extraction.colbert import retrieval_identity
            cache_config["colbert"] = retrieval_identity(self.feature_config.colbert)
        return cache_config

    def _spec_fingerprint(self, spec: ExplicitFeatureSpec) -> str:
        return json.dumps(self._cache_config([spec]), sort_keys=True, default=_json_default)

    def _resolve_vllm_model_inventory(self) -> Dict[str, str]:
        configured = self.feature_config.vllm_model_name
        mode = str(self.feature_config.vllm_mode or "server")
        server_urls = parse_server_urls(self.feature_config.vllm_server_url)
        if not _should_autodiscover_model_name(configured):
            return {url: str(configured) for url in server_urls}
        if mode != "server":
            raise ValueError(
                "vllm_model_name='auto' is only supported for vllm_mode='server'; "
                f"got vllm_mode={mode!r}. Provide an explicit model name for this mode."
            )
        if set(self._resolved_vllm_models_by_url) == set(server_urls):
            return dict(self._resolved_vllm_models_by_url)
        with OpenAIClientPool(
            server_urls=server_urls,
            api_key=getattr(self.feature_config, "vllm_api_key", "EMPTY"),
            timeout=getattr(self.feature_config, "extraction_request_timeout", 900.0),
            max_retries=0,
        ) as client_pool:
            self._resolved_vllm_models_by_url = {
                server_url: _discover_openai_compatible_model_name(
                    client_pool.client_for_url(server_url),
                    server_url=server_url,
                    purpose="explicit feature extraction",
                )
                for server_url in server_urls
            }
        return dict(self._resolved_vllm_models_by_url)

    def _resolve_vllm_model_name(self) -> str:
        if self._resolved_vllm_model_name is None:
            self._resolved_vllm_model_name = _endpoint_model_inventory_identity(
                self._resolve_vllm_model_inventory()
            )
        return self._resolved_vllm_model_name


class CodexCLIExplicitFeatureExtractionProvider(VLLMExplicitFeatureExtractionProvider):
    """Resumable explicit-feature provider backed by one `codex exec` call per note."""

    reads_complete_documents = True

    def _cache_config(self, specs: List[ExplicitFeatureSpec]) -> Dict[str, Any]:
        context_strategy = (
            str(getattr(self.feature_config, "extraction_context_strategy", "tail"))
            .strip()
            .lower()
            .replace("-", "_")
        )
        identity = {
            "features": [_spec_extraction_contract_dict(spec) for spec in specs],
            "prompt_template_version": EXTRACTION_PROMPT_VERSION,
            "extraction_provider": "codex_cli",
            "codex_cli_executable": getattr(
                self.feature_config,
                "codex_cli_executable",
                "codex",
            ),
            "codex_cli_model_name": _codex_optional_value(
                getattr(self.feature_config, "codex_cli_model_name", "gpt-5.4-mini")
            ),
            "codex_cli_reasoning_effort": _codex_optional_value(
                getattr(self.feature_config, "codex_cli_reasoning_effort", "medium")
            ),
            "codex_cli_extra_args": _codex_extra_args(
                getattr(self.feature_config, "codex_cli_extra_args", [])
            ),
            "complete_document": context_strategy == "tail",
            "extraction_max_text_length": self.feature_config.extraction_max_text_length,
            "extraction_grouping_strategy": getattr(
                self.feature_config,
                "extraction_grouping_strategy",
                "clinical_domain",
            ),
            "extraction_context_strategy": context_strategy,
            "extraction_context_compactor_version": CONTRACT_LEXICAL_CONTEXT_VERSION,
            "extraction_grouping_version": EXTRACTION_GROUPING_VERSION,
            "patient_text_hash": self._active_text_hash,
            "source_text_temporally_valid_by_design": bool(
                getattr(
                    self.feature_config,
                    "source_text_temporally_valid_by_design",
                    False,
                )
            ),
            "temporal_boundary_enforced": not bool(
                getattr(
                    self.feature_config,
                    "source_text_temporally_valid_by_design",
                    False,
                )
            ),
            "max_variables_per_extraction_request": int(
                getattr(self.feature_config, "max_variables_per_extraction_request", 10)
            ),
            "extraction_batch_size": int(self.feature_config.extraction_batch_size),
        }
        if context_strategy == "colbert":
            from ..extraction.colbert import retrieval_identity
            identity["colbert"] = retrieval_identity(self.feature_config.colbert)
        return identity

    def _extract_spec_group(
        self,
        dataset: pd.DataFrame,
        specs: List[ExplicitFeatureSpec],
    ) -> pd.DataFrame:
        logger.info(
            "Extracting %s agentic feature(s) with Codex CLI: %s",
            len(specs),
            [spec.name for spec in specs],
        )
        texts = dataset[self.config.text_column].fillna("").astype(str).tolist()
        max_workers = max(
            1,
            min(
                len(texts),
                int(getattr(self.feature_config, "codex_cli_parallelism", 4) or 4),
            ),
        )
        if max_workers == 1:
            results = [self._extract_one_text(text, specs) for text in texts]
        else:
            results = [None] * len(texts)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {
                    executor.submit(self._extract_one_text, text, specs): idx
                    for idx, text in enumerate(texts)
                }
                for future in as_completed(future_to_idx):
                    results[future_to_idx[future]] = future.result()
            results = [
                result if result is not None else _missing_values_for_specs(specs)
                for result in results
            ]
        return _extraction_results_to_dataframe(results, specs)

    def _extract_one_text(
        self,
        text: str,
        specs: List[ExplicitFeatureSpec],
    ) -> Dict[str, Any]:
        context_strategy = (
            str(getattr(self.feature_config, "extraction_context_strategy", "tail"))
            .strip()
            .lower()
            .replace("-", "_")
        )
        trusted_temporal_source = bool(
            getattr(
                self.feature_config,
                "source_text_temporally_valid_by_design",
                False,
            )
        )
        if context_strategy == "colbert":
            from ..extraction.colbert import get_retriever
            text = get_retriever(self.feature_config.colbert).retrieve(
                text, specs, top_k=self.feature_config.colbert.top_k)["context"]
            if (self.feature_config.extraction_max_text_length is not None
                    and len(text) > self.feature_config.extraction_max_text_length):
                raise ValueError("ColBERT retrieved context exceeds extraction_max_text_length")
            context_strategy = "tail"
        base_prompt = _codex_extraction_prompt(
            text,
            specs,
            max_text_length=(
                self.feature_config.extraction_max_text_length
                if context_strategy == "contract_lexical_rag"
                else None
            ),
            context_strategy=context_strategy,
            source_text_temporally_valid_by_design=trusted_temporal_source,
        )
        prompt = base_prompt
        best_result = None
        max_attempts = max(1, int(self.feature_config.extraction_max_retries))
        for attempt in range(max_attempts):
            try:
                response = _run_codex_exec(
                    prompt,
                    executable=getattr(self.feature_config, "codex_cli_executable", "codex"),
                    model_name=getattr(
                        self.feature_config,
                        "codex_cli_model_name",
                        "gpt-5.4-mini",
                    ),
                    reasoning_effort=getattr(
                        self.feature_config,
                        "codex_cli_reasoning_effort",
                        "medium",
                    ),
                    extra_args=getattr(self.feature_config, "codex_cli_extra_args", []),
                    timeout=getattr(
                        self.feature_config,
                        "extraction_request_timeout",
                        900.0,
                    ),
                )
                parsed = _parse_extraction_response_with_issues(
                    response.content,
                    specs,
                )
                result = parsed.values
                if best_result is None or sum(
                    1 for value in result.values() if not value.is_missing
                ) > sum(1 for value in best_result.values() if not value.is_missing):
                    best_result = result
                if not parsed.issues:
                    return result
                if attempt < max_attempts - 1:
                    logger.warning(
                        "Codex CLI extraction response had schema issues on attempt "
                        "%s/%s: %s. Requesting repaired JSON.",
                        attempt + 1,
                        max_attempts,
                        "; ".join(parsed.issues[:3]),
                    )
                    prompt = (
                        f"{base_prompt}\n\nPrevious Codex response:\n{response.content}\n\n"
                        f"{build_extraction_repair_prompt(parsed.issues, specs, source_text_temporally_valid_by_design=trusted_temporal_source)}"
                    )
                    continue
            except Exception as exc:
                if attempt >= max_attempts - 1:
                    logger.warning(
                        "Codex CLI extraction failed after %s attempt(s): %s",
                        attempt + 1,
                        exc,
                    )
                    break
                delay = retry_delay(
                    attempt,
                    initial_delay=getattr(
                        self.feature_config,
                        "extraction_retry_initial_delay",
                        1.0,
                    ),
                    max_delay=getattr(
                        self.feature_config,
                        "extraction_retry_max_delay",
                        30.0,
                    ),
                    backoff_factor=getattr(
                        self.feature_config,
                        "extraction_retry_backoff_factor",
                        2.0,
                    ),
                )
                logger.warning(
                    "Codex CLI extraction failed on attempt %s/%s with %s: %s. "
                    "Retrying in %.2fs.",
                    attempt + 1,
                    max_attempts,
                    exc.__class__.__name__,
                    exc,
                    delay,
                )
                import time

                time.sleep(delay)

        return best_result if best_result is not None else _missing_values_for_specs(specs)


def _extraction_results_to_dataframe(
    results: Sequence[Dict[str, Any]],
    specs: List[ExplicitFeatureSpec],
) -> pd.DataFrame:
    data: Dict[str, List[Any]] = {}
    for spec in specs:
        values = []
        missing_flags = []
        for result in results:
            val = result.get(spec.name)
            if val:
                values.append(val.value)
                missing_flags.append(val.is_missing)
            else:
                values.append(None)
                missing_flags.append(True)
        data[f"explicit_feat_{spec.name}"] = values
        data[f"explicit_feat_{spec.name}_missing"] = missing_flags
    return pd.DataFrame(data)


def make_explicit_feature_extraction_provider(
    *,
    config: AppliedInferenceConfig,
    output_dir: Path,
) -> Any:
    if _codex_cli_provider(getattr(config.explicit_features, "extraction_provider", "openai")):
        return CodexCLIExplicitFeatureExtractionProvider(config=config, output_dir=output_dir)
    return VLLMExplicitFeatureExtractionProvider(config=config, output_dir=output_dir)


class CausalForestExplicitEvaluator:
    """Fit/evaluate one explicit-feature causal forest split."""

    def __init__(
        self,
        config: AppliedInferenceConfig,
        cf_config: ExplicitFeatureForestConfig,
    ):
        self.config = config
        self.cf_config = cf_config

    def evaluate_split(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        specs: List[ExplicitFeatureSpec],
        fold_id: int,
    ) -> SplitEvaluation:
        train_T = np.asarray(train_df[self.config.treatment_column].values).flatten()
        train_Y = np.asarray(train_df[self.config.outcome_column].values).flatten()
        test_T = np.asarray(test_df[self.config.treatment_column].values).flatten()
        test_Y = np.asarray(test_df[self.config.outcome_column].values).flatten()

        X_train, W_train, x_names, w_names, means, stds = _build_features(train_df, specs)
        X_test, W_test, _, _, _, _ = _build_features(test_df, specs, means, stds)
        actual_x_dim = 0 if X_train is None else X_train.shape[1]
        if X_train is None:
            X_train = np.zeros((len(train_df), 1), dtype=np.float32)
            X_test = np.zeros((len(test_df), 1), dtype=np.float32)
            x_names = ["intercept_effect"]

        nuisance_train = _hstack_present(X_train, W_train)
        nuisance_test = _hstack_present(X_test, W_test)
        if nuisance_train is None or nuisance_test is None:
            raise ValueError("Unable to build explicit-feature nuisance matrices")

        forest = CausalForestHead(
            n_estimators=self.cf_config.n_estimators,
            max_depth=self.cf_config.max_depth,
            min_samples_leaf=self.cf_config.min_samples_leaf,
            max_features=self.cf_config.max_features,
            honest=self.cf_config.honest,
            inference=self.cf_config.inference,
            random_state=42 + fold_id,
            outcome_type=self.config.outcome_type,
        )
        forest.fit(X_train, train_T, train_Y, W=W_train)
        cf_preds = forest.predict(X_test, return_ci=True)
        tau = cf_preds["tau_pred"]

        propensity = _fit_predict_propensity(
            nuisance_train,
            train_T,
            nuisance_test,
            self.cf_config,
            random_state=142 + fold_id,
        )
        outcome_pred = _fit_predict_outcome(
            nuisance_train,
            train_Y,
            nuisance_test,
            self.config.outcome_type,
            self.cf_config,
            random_state=242 + fold_id,
        )

        y0_prob = outcome_pred - propensity * tau
        y1_prob = outcome_pred + (1.0 - propensity) * tau
        if self.config.outcome_type == "binary":
            y0_prob = np.clip(y0_prob, 0, 1)
            y1_prob = np.clip(y1_prob, 0, 1)

        predictions = test_df.copy()
        predictions["pred_ite_prob"] = tau
        predictions["pred_y0_prob"] = y0_prob
        predictions["pred_y1_prob"] = y1_prob
        predictions["pred_propensity_prob"] = propensity
        predictions["pred_outcome_prob"] = outcome_pred
        predictions["cv_fold"] = fold_id
        if "tau_lower" in cf_preds:
            predictions["pred_ite_lower"] = cf_preds["tau_lower"]
            predictions["pred_ite_upper"] = cf_preds["tau_upper"]

        metrics = {
            "fold": fold_id,
            "n_train": len(train_df),
            "n_test": len(test_df),
            "n_explicit_features": len(specs),
            "n_x_features": actual_x_dim,
            "n_w_features": 0 if W_train is None else W_train.shape[1],
            "ate_estimate": float(np.mean(tau)),
            "r_loss": float(_r_loss(test_Y, test_T, outcome_pred, propensity, tau)),
            "treatment_auroc": _safe_roc_auc(test_T, propensity),
            "outcome_auroc": (
                _safe_roc_auc(test_Y, outcome_pred)
                if self.config.outcome_type == "binary"
                else None
            ),
            "x_feature_names": x_names,
            "w_feature_names": w_names,
        }
        if "true_ite_prob" in test_df.columns:
            metrics["oracle_true_ite_corr"] = _safe_corr(test_df["true_ite_prob"].values, tau)
            metrics["oracle_true_ite_mae"] = float(
                np.mean(np.abs(np.asarray(test_df["true_ite_prob"].values) - tau))
            )

        return SplitEvaluation(predictions=predictions, metrics=metrics)


class StructuredInteractionExplicitEvaluator:
    """Nested-CV structured S-learner with treatment interactions.

    The outer fit/test boundary is supplied by the calling runner.  Within the
    fit frame, :class:`StructuredInteractionHead` selects regularization using
    observed outcome loss only.  The held-out frame is scored once after that
    choice and final fit are frozen.
    """

    def __init__(
        self,
        config: AppliedInferenceConfig,
        cf_config: ExplicitFeatureForestConfig,
    ) -> None:
        self.config = config
        self.cf_config = cf_config

    def evaluate_split(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        specs: List[ExplicitFeatureSpec],
        fold_id: int,
    ) -> SplitEvaluation:
        train_t = np.asarray(
            train_df[self.config.treatment_column].values,
            dtype=float,
        ).reshape(-1)
        train_y = np.asarray(
            train_df[self.config.outcome_column].values,
            dtype=float,
        ).reshape(-1)
        test_t = np.asarray(
            test_df[self.config.treatment_column].values,
            dtype=float,
        ).reshape(-1)
        test_y = np.asarray(
            test_df[self.config.outcome_column].values,
            dtype=float,
        ).reshape(-1)

        x_train, w_train, x_names, w_names, means, stds = _build_features(
            train_df,
            specs,
        )
        x_test, w_test, _, _, _, _ = _build_features(
            test_df,
            specs,
            means,
            stds,
        )
        actual_x_dim = 0 if x_train is None else int(x_train.shape[1])
        if x_train is None:
            # Keep an explicit zero column so a confounder-only registry is a
            # valid main-effect model with a constant treatment effect.
            x_train = np.zeros((len(train_df), 1), dtype=np.float32)
            x_test = np.zeros((len(test_df), 1), dtype=np.float32)
            x_names = ["intercept_effect"]

        features_train = _hstack_present(x_train, w_train)
        features_test = _hstack_present(x_test, w_test)
        if features_train is None or features_test is None:
            raise ValueError("Unable to build explicit-feature interaction matrices")

        interact_all = bool(getattr(self.cf_config, "interaction_interact_all_features", True))
        head = StructuredInteractionHead(
            outcome_type=self.config.outcome_type,
            regularization_grid=getattr(
                self.cf_config,
                "interaction_regularization_grid",
                (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0),
            ),
            inner_folds=int(getattr(self.cf_config, "interaction_inner_folds", 3)),
            interact_all_features=interact_all,
            random_state=42 + int(fold_id),
            max_iter=int(getattr(self.cf_config, "interaction_max_iter", 3000)),
        )
        modifier_indices = None if interact_all else np.arange(x_train.shape[1])
        head.fit(
            features_train,
            train_t,
            train_y,
            modifier_indices=modifier_indices,
        )
        y0, y1 = head.predict_potential_outcomes(features_test)
        tau = y1 - y0
        outcome_prediction = head.predict_observed_outcome(features_test, test_t)

        propensity = _fit_predict_propensity(
            features_train,
            train_t,
            features_test,
            self.cf_config,
            random_state=142 + int(fold_id),
        )
        # Prediction artifacts are fail-closed against accidental synthetic
        # oracle propagation even when a lower-level caller forgets to strip it.
        prediction_input = test_df.drop(
            columns=[column for column in test_df.columns if str(column).startswith("true_")],
            errors="ignore",
        )
        predictions = prediction_input.copy()
        predictions["pred_ite_prob"] = tau
        predictions["pred_y0_prob"] = y0
        predictions["pred_y1_prob"] = y1
        predictions["pred_propensity_prob"] = propensity
        predictions["pred_outcome_prob"] = outcome_prediction
        predictions["cv_fold"] = int(fold_id)

        tuning = head.tuning_result_
        assert tuning is not None
        metrics = {
            "fold": int(fold_id),
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "n_explicit_features": int(len(specs)),
            "n_x_features": int(actual_x_dim),
            "n_w_features": 0 if w_train is None else int(w_train.shape[1]),
            "n_interaction_features": int(len(head.modifier_indices_)),
            "interact_all_features": interact_all,
            "ate_estimate": float(np.mean(tau)),
            "r_loss": float(_r_loss(test_y, test_t, outcome_prediction, propensity, tau)),
            "treatment_auroc": _safe_roc_auc(test_t, propensity),
            "outcome_auroc": (
                _safe_roc_auc(test_y, outcome_prediction)
                if self.config.outcome_type == "binary"
                else None
            ),
            "x_feature_names": x_names,
            "w_feature_names": w_names,
            "effect_estimator": "interaction_s_learner",
            "interaction_selected_regularization": float(tuning.selected_regularization),
            "interaction_selection_metric": tuning.selection_metric,
            "interaction_inner_folds": int(tuning.n_splits),
            "interaction_mean_validation_loss": {
                str(key): float(value) for key, value in tuning.mean_validation_loss.items()
            },
        }
        return SplitEvaluation(predictions=predictions, metrics=metrics)


def build_agent_prompt(
    context: Dict[str, Any],
    search_config: AgenticFeatureSearchConfig,
) -> str:
    """Construct the proposal prompt sent to the LLM agent."""
    from .all_evidence_fusion import FUSION_PROMPT_VERSION

    if context.get("prompt_version") == FUSION_PROMPT_VERSION:
        from .all_evidence_fusion import render_all_evidence_fusion_context_prompt

        return render_all_evidence_fusion_context_prompt(context)
    from .all_evidence_post_extraction_review import (
        POST_EXTRACTION_REVIEW_PROMPT_VERSION,
        render_post_extraction_review_prompt,
    )

    if context.get("prompt_version") == POST_EXTRACTION_REVIEW_PROMPT_VERSION:
        return render_post_extraction_review_prompt(context)
    if context.get("prompt_version") == "neural_query_feature_v1":
        from .neural_query_agentic_forest import render_query_feature_prompt

        return render_query_feature_prompt(context)
    if context.get("prompt_version") == "neural_query_registry_v1":
        from .neural_query_agentic_forest import render_query_registry_prompt

        return render_query_registry_prompt(context)
    if context.get("prompt_version") == "neural_query_review_v1":
        from .neural_query_agentic_forest import render_query_review_prompt

        return render_query_review_prompt(context)
    if context.get("prompt_version") in {
        "tfidf_topic_label_v2",
        "tfidf_topic_recovery_v2",
        "tfidf_orphan_ngram_label_v1",
    }:
        from .tfidf_topic_agentic_forest import render_topic_label_prompt

        return render_topic_label_prompt(context)
    if context.get("prompt_version") == "tfidf_topic_name_harmonization_v2":
        from .tfidf_topic_agentic_forest import render_topic_name_harmonization_prompt

        return render_topic_name_harmonization_prompt(context)
    if context.get("prompt_version") == "tfidf_topic_global_dedup_v2":
        from .tfidf_topic_agentic_forest import render_topic_global_dedup_prompt

        return render_topic_global_dedup_prompt(context)
    if context.get("prompt_version") in {
        "tfidf_topic_value_harmonization_v2",
        "tfidf_topic_value_repair_v2",
    }:
        from .tfidf_topic_agentic_forest import render_topic_value_harmonization_prompt

        return render_topic_value_harmonization_prompt(context)
    if context.get("prompt_version") == "agentic_attention_consensus_disambiguation_v1":
        return build_attention_consensus_disambiguation_prompt(context, search_config)

    if context.get("prompt_version") == "agentic_attention_consensus_recovery_v1":
        return build_attention_consensus_recovery_prompt(context, search_config)

    if context.get("prompt_version") == "multi_model_agentic_alias_resolution_v1":
        return build_multi_model_agentic_alias_resolution_prompt(context, search_config)

    if context.get("prompt_version") == "multi_model_agentic_value_harmonization_v1":
        return build_multi_model_agentic_value_harmonization_prompt(context, search_config)

    if context.get("prompt_version") == "multi_model_agentic_concept_inventory_v1":
        return build_multi_model_agentic_concept_inventory_prompt(context, search_config)

    if context.get("prompt_version") in {
        "multi_model_agentic_cluster_labeling_v1",
        "multi_model_agentic_cluster_labeling_v2",
    }:
        return build_multi_model_agentic_cluster_labeling_prompt(context, search_config)

    if context.get("prompt_version") == _PARSIMONY_FACTOR_PROMPT_VERSION:
        return build_multi_model_agentic_parsimony_factor_prompt(context, search_config)

    if context.get("prompt_version") == "multi_model_agentic_consistency_v1":
        return build_multi_model_agentic_consistency_prompt(context, search_config)

    if context.get("prompt_version") == "multi_model_agentic_extracted_feature_review_v1":
        return build_multi_model_agentic_extracted_feature_review_prompt(context, search_config)

    if context.get("prompt_version") == "agentic_attention_variable_forest_v1":
        return build_attention_variable_agent_prompt(context, search_config)

    if context.get("prompt_version") == "multi_model_agentic_forest_v1":
        return build_multi_model_agentic_forest_prompt(context, search_config)

    if context.get("prompt_version") == "multi_model_agentic_evidence_digest_role_v1":
        return build_multi_model_evidence_digest_role_prompt(context, search_config)

    if context.get("search_mode") == "broad_screen":
        if context.get("broad_screen_stage") == "selection":
            return build_broad_selection_agent_prompt(context, search_config)
        return build_broad_agent_prompt(context, search_config)

    context_json = json.dumps(context, indent=2, default=_json_default)
    current_feature_count = len(context.get("current_features", []))
    feature_status = (
        "No variables are currently included; propose an initial variable to extract."
        if current_feature_count == 0
        else (
            "The current variables are already included; propose additions, "
            "removals, or role updates only when the nested-CV context and "
            "prior feedback justify them."
        )
    )
    return f"""You are helping design a causal inference feature set for a causal forest.

{feature_status}
Propose variables that are plausibly extractable from the text and could improve confounding adjustment or CATE heterogeneity.
Use the clinical question, estimand metadata, nested-CV context, extraction summaries, clinical text examples, and prior feedback to decide which variables are worth trying. Define each extraction target precisely enough that the downstream feature extractor can operationalize it.
Candidate feedback may include role_diagnostics from train-fold regressions adjusted for current confounders. Treat a candidate with both treatment and outcome association as a likely confounder. Treat a candidate with treatment-by-candidate interaction evidence in the outcome model as a likely effect modifier. Use those diagnostics to revise roles or extraction targets when a prior candidate was rejected.

Return JSON only with this shape:
{{
  "proposals": [
    {{
      "action": "add|remove|update_role|none",
      "name": "snake_case_variable_name",
      "type": "categorical|continuous",
      "categories": ["category_a", "category_b"],
      "roles": ["confounder", "effect_modifier"],
      "description": "exact extraction target",
      "rationale": "why this may help",
      "expected_signal": "treatment, outcome, or tau signal expected"
    }}
  ]
}}

Limits:
- At most {search_config.max_additions_per_iter} add proposals.
- At most {search_config.max_removals_per_iter} remove proposals.
- Use "none" if no defensible variable is available.
- For categorical variables, provide 2-8 mutually exclusive categories.
- Review iteration_feedback and recent_decisions before proposing. Do not repeat
  a rejected feature unchanged; if revisiting a rejected concept, change the
  extraction target, type/categories, or role to directly address failed_checks.

Current nested-CV context:
{context_json}
"""


def build_attention_variable_agent_prompt(
    context: Dict[str, Any],
    search_config: AgenticFeatureSearchConfig,
) -> str:
    """Construct the proposal prompt for attention-evidence variable discovery."""
    context_json = json.dumps(context, indent=2, default=_json_default)
    max_proposals = int(
        context.get(
            "max_proposals",
            max(1, int(getattr(search_config, "max_additions_per_iter", 3))),
        )
    )
    return f"""You are inspecting attention evidence from a neural text model used in a downstream clinical prediction task.

Your task is narrow: propose at most {max_proposals} explicit pre-treatment patient-level variables that the neural model may be capturing. Each evidence item contains highly attended token spans inside highly attended clinical text snippets. Treat those token spans as the primary clue; use the surrounding evidence_snippet text to name the variable loosely but concretely. If the attended evidence does not suggest a reusable extractable patient-level variable, return a single "none" proposal.

Rules:
- Use only pre-treatment information visible in the attended evidence.
- Before choosing proposals, mentally inventory repeated high-attention spans and the patient-level fields they imply; mundane patient-level fields count if they repeatedly appear with high token salience.
- Prefer variables whose values appear repeatedly in top_token_spans, attended_token_summary, or the surrounding evidence_snippet and look extractable across many patients.
- Avoid sparse one-off concepts, downstream treatment response, toxicity after treatment, survival, and outcome-derived variables.
- Avoid aliases or near-duplicates of current_features and excluded_feature_names.
- If rejected_low_coverage_features is non-empty, do not repeat those extraction targets unchanged; propose a broader or more directly documented target only if the attended chunks support it.
- If rejected_low_signal_features or multivariable_signal_feedback indicate weak downstream prediction signal, propose different repeated variables suggested by the attended evidence.
- For every add proposal, name the specific high-attention token span or phrase from the evidence in the rationale.

Return JSON only with this shape:
{{
  "proposals": [
    {{
      "action": "add|none",
      "name": "snake_case_variable_name",
      "type": "categorical|continuous",
      "categories": ["category_a", "category_b"],
      "description": "exact pre-treatment extraction target",
      "rationale": "which attended token span or phrase supports this variable",
      "expected_signal": "briefly state what downstream signal this variable might carry, if any"
    }}
  ]
}}

Limits:
- At most {max_proposals} add proposals.
- Use "none" when the attended chunks do not support a defensible extractable variable.
- For categorical variables, provide 2-8 mutually exclusive categories.
- Use distinct names; avoid near-duplicate aliases for the same concept.

Current attention-evidence context:
{context_json}
"""


def build_multi_model_evidence_digest_role_prompt(
    context: Dict[str, Any],
    search_config: AgenticFeatureSearchConfig,
) -> str:
    """Render flattened evidence blurbs as a simple concept proposal prompt."""
    del search_config
    max_proposals = int(context.get("max_proposals") or 30)
    blurbs = [str(item).strip() for item in (context.get("text_blurbs") or []) if str(item).strip()]
    blurbs_text = (
        "\n\n".join(f"{idx}. {blurb}" for idx, blurb in enumerate(blurbs, start=1))
        if blurbs
        else "No usable text blurbs were supplied."
    )
    return f"""You are reviewing text blurbs that emerged from clinical predictive modeling.

The prediction target and downstream scientific task are intentionally hidden. Do not infer or discuss them.

Identify patient-level concepts that could explain why these blurbs repeatedly emerged. Convert noisy phrases, fragments, and snippets into reusable variables a downstream extractor could read from a clinical note.

Return a fairly broad list: up to {max_proposals} concepts. It is better to include plausible distinct concepts than to be overly sparse; a later pipeline will extract, merge aliases, and validate them.

Rules:
- Propose concepts/variables, not raw tokens or copied phrases.
- Ignore patient names, identifiers, accession numbers, note boilerplate, and document-section labels.
- Keep each concept precise enough for extraction from a full clinical note.
- Use categorical type when a small set of values is natural; use continuous type for numeric measurements.
- If the blurbs support no reusable patient-level concept, return one proposal with action "none".

Return JSON only:
{{
  "proposals": [
    {{
      "action": "add|none",
      "name": "snake_case_variable_name",
      "type": "categorical|continuous",
      "categories": ["category_a", "category_b"],
      "description": "exact extraction target",
      "rationale": "which blurb numbers support this concept",
      "expected_signal": "brief description of the repeated text pattern"
    }}
  ]
}}

Text blurbs:
{blurbs_text}
"""


def build_attention_consensus_disambiguation_prompt(
    context: Dict[str, Any],
    search_config: AgenticFeatureSearchConfig,
) -> str:
    """Construct the alias-resolution prompt for attention-variable consensus."""
    context_json = json.dumps(context, indent=2, default=_json_default)
    threshold = int(context.get("consensus_threshold", 2))
    return f"""You are resolving aliases among candidate explicit variables proposed from separate inner folds.

Your task is narrow: merge aliases only. Group proposal names only when they refer to the same explicit pre-treatment patient-level extraction target. Do not create new variables, broaden a target, or merge related but clinically distinct concepts.

Return JSON only with this shape:
{{
  "groups": [
    {{
      "canonical_name": "one_existing_snake_case_member_name",
      "member_names": ["existing_proposal_name_a", "existing_proposal_name_b"],
      "member_folds": [1, 2],
      "type": "categorical|continuous",
      "categories": ["category_a", "category_b"],
      "description": "exact shared extraction target",
      "rationale": "brief reason these names are aliases"
    }}
  ],
  "unmerged": [
    {{
      "name": "existing_proposal_name",
      "reason": "why this was not equivalent to another proposal"
    }}
  ]
}}

Rules:
- Use only names that appear in proposed_variables_by_fold.
- canonical_name must be one of the member_names.
- A group can pass consensus only if its members appear in at least {threshold} distinct folds.
- Merge exact aliases such as different names for the same measurement target and timing.
- Do not merge variables with different types or incompatible categorical categories.
- Do not merge nearby but distinct clinical concepts, such as baseline value versus change over time, biomarker status versus mutation burden, or disease stage versus metastatic site.
- Leave single-fold or ambiguous concepts unmerged rather than forcing a group.

Current consensus-disambiguation context:
{context_json}
"""


def build_attention_consensus_recovery_prompt(
    context: Dict[str, Any],
    search_config: AgenticFeatureSearchConfig,
) -> str:
    """Construct the stability/recovery prompt for attention-variable consensus."""
    del search_config
    context_json = json.dumps(context, indent=2, default=_json_default)
    max_selected = int(context.get("max_selected_candidates", 6))
    return f"""You are selecting stable explicit variables for a causal forest from candidates proposed by a neural attention-based feature extractor across separate inner folds.

The outer test fold is not included here. All evidence comes from the current outer-train data only.

Your goals:
- Keep candidates that passed the fold-consensus gate unless they are redundant, likely leakage, or clinically not extractable.
- Recover plausible real variables that missed one or more folds when the missing-fold pattern looks unstable rather than truly absent.
- Preserve honest causal inference: use only the supplied candidate_summaries and do not invent new variables.

Return JSON only with this shape:
{{
  "proposals": [
    {{
      "action": "add|none",
      "name": "existing_candidate_name",
      "type": "categorical|continuous",
      "categories": ["category_a", "category_b"],
      "roles": ["confounder", "effect_modifier"],
      "description": "exact pre-treatment extraction target",
      "rationale": "why this candidate is stable enough or should be recovered",
      "expected_signal": "treatment, outcome, attention, or pseudo-outcome signal expected"
    }}
  ]
}}

Rules:
- Choose only names listed in candidate_summaries.
- Prefer candidates that pass the fold-consensus gate.
- You may recover a below-threshold candidate only when its supporting folds have coherent names, roles, descriptions, and rationales, and the missing folds look like attention/proposal instability.
- Do not select variables that are post-treatment, outcome-derived, treatment choice itself, response, survival, or toxicity.
- Keep roles tied to evidence: nuisance/treatment+outcome support implies confounder; R-stage, residual, interaction, or pseudo-outcome support implies effect_modifier; both signals may justify both roles.
- Return at most {max_selected} add proposals.

Current attention-consensus recovery context:
{context_json}
"""


def build_multi_model_agentic_alias_resolution_prompt(
    context: Dict[str, Any],
    search_config: AgenticFeatureSearchConfig,
) -> str:
    """Construct a generic alias-resolution prompt for multi-model proposals."""
    del search_config
    context_json = json.dumps(context, indent=2, default=_json_default)
    return f"""You are resolving aliases among explicit patient-level variables proposed for causal inference.

Your task is narrow: merge aliases only. Group names only when they refer to the same exact pre-treatment extraction target. Do not create new variables, broaden a target, or merge related but clinically distinct concepts.

Return JSON only with this shape:
{{
  "groups": [
    {{
      "canonical_name": "one_existing_candidate_or_known_name",
      "member_names": ["existing_name_a", "existing_name_b"],
      "type": "categorical|continuous",
      "categories": ["category_a", "category_b"],
      "description": "exact shared extraction target",
      "roles": ["confounder", "effect_modifier"],
      "rationale": "brief reason these names are aliases"
    }}
  ],
  "unmerged": [
    {{
      "name": "existing_name",
      "reason": "why this was not equivalent to another proposal"
    }}
  ]
}}

Rules:
- Use only names that appear in proposed_features or known_canonical_features.
- canonical_name must be one of member_names, unless a member is an alias of a known_canonical_features name; in that case prefer the known canonical name.
- Merge only when the names, descriptions, categories, and roles describe the same measurable variable.
- Do not merge broad and narrow targets, timing variants, measurement-method variants, or clinically related but distinct concepts.
- Leave singletons and ambiguous concepts unmerged.

Current alias-resolution context:
{context_json}
"""


def build_multi_model_agentic_value_harmonization_prompt(
    context: Dict[str, Any],
    search_config: AgenticFeatureSearchConfig,
) -> str:
    """Construct a prompt that canonicalizes value contracts for selected features."""
    del search_config
    context_json = json.dumps(context, indent=2, default=_json_default)
    return f"""You are harmonizing value contracts for explicit patient-level variables before extraction.

Your task is narrow: keep the same variables, but make each variable's extracted values canonical and machine-usable. This applies to categorical variables and to variables that look numeric; numeric variables must not accept strings like "unknown", "high", or "not reported" as values.

Return JSON only with this shape:
{{
  "features": [
    {{
      "name": "existing_variable_name",
      "type": "categorical|continuous",
      "categories": ["canonical_category_a", "canonical_category_b"],
      "description": "exact extraction target and value policy",
      "value_aliases": {{
        "canonical_category_a": ["alias_a", "alias_b"]
      }},
      "missing_values": ["unknown", "not_reported"],
      "rationale": "brief reason for the value contract"
    }}
  ]
}}

Rules:
- Include every selected feature exactly once. Do not add, remove, or rename variables.
- Preserve each variable's causal roles conceptually; this pass only harmonizes values.
- For continuous variables, set "type" to "continuous" and omit categories or set them to null/[].
- For continuous variables, the description must say to return a numeric value only and return null for unknown, not reported, not assessed, qualitative-only values such as high/low, or unavailable values.
- For categorical variables, choose mutually exclusive canonical categories and list only those categories.
- Use value_aliases to map synonymous or formatting variants to canonical categories.
- Do not include missing-like values such as unknown, not_reported, not_assessed, not_tested, unavailable, or null as categories; represent unavailable values as null.
- Prefer clinically meaningful thresholds over broad high/low categories when thresholds are available in the candidate category list or feature evidence.

Current value-harmonization context:
{context_json}
"""


def build_multi_model_agentic_concept_inventory_prompt(
    context: Dict[str, Any],
    search_config: AgenticFeatureSearchConfig,
) -> str:
    """Construct the source-grounded concept inventory prompt."""
    del search_config
    context_json = json.dumps(
        context,
        separators=(",", ":"),
        default=_json_default,
    )
    max_concepts = int(context.get("max_concepts", 60))
    return f"""You are analyzing evidence from clinical-note text models.

Task: identify patient-level concepts that appear to be represented in common
among the evidence families supplied in the context.

Evidence families:
- bow_feature_evidence: sparse text features and repeated phrase signals.
- embedding_retrieved_text_evidence: retrieved real text chunks and concept scores.
- htr_attended_text_evidence: tokens or spans highlighted by neural text models.

Look for concepts that recur as the same patient-level field across at least two
evidence families when possible. Concepts may be ordinary chart fields, clinical
history, baseline lab values, demographics, staging, biomarkers, symptoms,
performance status, comorbidities, treatment history, social history, or other
extractable note-level fields.

Do not rank concepts by downstream modeling usefulness. Do not ignore mundane
or header-like variables when they are repeatedly represented. Collapse aliases
that refer to the same field, and keep distinct fields separate.

Return at most {max_concepts} concepts. Return JSON only with this shape:
{{
  "concepts": [
    {{
      "name": "snake_case_concept_name",
      "label": "short human-readable label",
      "value_kind": "binary|categorical|continuous|ordinal|text|unknown",
      "source_families": ["bow", "embedding_contrast", "htr"],
      "source_overlap": 2,
      "supporting_phrases": ["short phrase or token evidence"],
      "example_values_or_phrases": ["example values or note phrases"],
      "extractability": "high|medium|low",
      "notes": "brief source-grounded explanation"
    }}
  ],
  "omitted_uncertain": ["optional short labels for weak one-source concepts"]
}}

Current concept-inventory context:
{context_json}
"""


def build_multi_model_agentic_cluster_labeling_prompt(
    context: Dict[str, Any],
    search_config: AgenticFeatureSearchConfig,
) -> str:
    """Construct the prompt for labeling deterministic evidence clusters."""
    del search_config
    context_json = json.dumps(
        context,
        separators=(",", ":"),
        default=_json_default,
    )
    max_concepts = int(context.get("max_concepts", 60))
    return f"""You are labeling clusters of clinical-note text evidence.

The clusters were generated upstream from sparse text features, retrieved text
chunks, and attended text snippets. They were not generated from a clinical
dictionary. Your job is to turn these supplied clusters into a compact inventory
of patient-level concepts. A cluster is evidence, not a one-concept container:
one cluster may support multiple distinct concepts, and one concept may be
supported by multiple clusters.

Task:
- Label clusters that represent extractable patient-level fields.
- Merge multiple clusters when they clearly refer to the same field.
- Split compound clusters when they contain multiple separately extractable
  patient-level fields. It is valid for several concepts to cite the same
  cluster_id.
- For mixed panels or grouped chart content such as CBCs, CMPs, vitals,
  demographics, molecular panels, pathology/IHC, or medication-history groups,
  consider the individual represented components instead of only the broad
  panel label.
- Reject clusters that are boilerplate, document structure, too broad, too
  noisy, or not patient-level.
- Use only the supplied clusters and evidence phrases. Do not invent concepts
  outside these clusters.
- Do not decide downstream modeling roles.

Return at most {max_concepts} concepts. Return JSON only with this shape:
{{
  "concepts": [
    {{
      "name": "snake_case_concept_name",
      "label": "short human-readable label",
      "value_kind": "binary|categorical|continuous|ordinal|text|unknown",
      "source_families": ["bow", "embedding_contrast", "htr"],
      "source_overlap": 2,
      "supporting_phrases": ["short phrase or token evidence"],
      "example_values_or_phrases": ["example values or note phrases"],
      "extractability": "high|medium|low",
      "cluster_ids": ["cluster_001", "cluster_014"],
      "notes": "brief source-grounded explanation"
    }}
  ],
  "rejected_clusters": [
    {{
      "cluster_id": "cluster_009",
      "reason": "brief reason this cluster was not a patient-level concept"
    }}
  ]
}}

Current cluster-labeling context:
{context_json}
"""


def build_multi_model_agentic_parsimony_factor_prompt(
    context: Dict[str, Any],
    search_config: AgenticFeatureSearchConfig,
) -> str:
    """Construct the post-extraction cluster-to-factor parsimony prompt."""
    del search_config
    context_json = json.dumps(
        context,
        separators=(",", ":"),
        default=_json_default,
    )
    max_factors = int(context.get("max_factors", 2))
    return f"""You are reviewing one empirically clustered group of structured clinical variables for parsimonious causal modeling.

The cluster was formed from the actual extracted patient-level values in the
current outer-training fold, supplemented by semantic similarity between the
variable contracts. The outer test fold is absent. Your task is conceptual:
decide whether up to {max_factors} operational patient-level factors can replace
at least two replaceable cluster members while preserving the information in
the group.

An underlying factor may be implicit rather than literally named in the note.
For example, several functional, symptom, and nutritional measurements might
support an operational frailty factor. An implicit factor is valid only when
you define reproducible evidence rules. The downstream extractor will read the
complete permitted note and must return null when the minimum evidence is not
present. Do not manufacture a value from general clinical intuition.

Rules:
- Use only member names listed in replaceable_members in `replaces`.
- Never replace a member listed in protected_members.
- Return at most {max_factors} factors and fewer factors than replaced members.
- The factors must collectively cover every role in required_role_union.
- Implicit factors must be categorical with 2-8 mutually exclusive categories.
- A continuous factor is allowed only for a directly interpretable measurement
  with an explicit unit; set inference_kind to `direct`.
- For every factor, give supporting indicators, contrary indicators, the
  minimum evidence needed for a non-null value, and a null policy.
- All evidence used to assign the factor must be temporally available before the treatment decision.
  Do not impose an additional ban based on whether the
  concept concerns treatment, response, prognosis, survival, or toxicity.
- Choose `retain_cluster` when the group is multidimensional, incoherent, not
  inferable reproducibly, or cannot be compressed without losing meaning.

Return JSON only with this shape:
{{
  "cluster_id": "cluster_001",
  "decision": "replace_cluster|retain_cluster",
  "replaces": ["existing_member_a", "existing_member_b"],
  "factors": [
    {{
      "name": "snake_case_factor_name",
      "inference_kind": "implicit|direct",
      "type": "categorical|continuous",
      "categories": ["category_a", "category_b"],
      "unit": "measurement unit for continuous factors or null",
      "roles": ["confounder", "effect_modifier"],
      "description": "precise patient-level factor",
      "supporting_indicators": ["specific note evidence supporting a value"],
      "contrary_indicators": ["specific evidence arguing against that value"],
      "minimum_evidence": "minimum evidence required for non-null extraction",
      "null_policy": "when the extractor must return null",
      "rationale": "why this factor represents the replaced value cluster"
    }}
  ],
  "rationale": "why replacement is or is not conceptually defensible"
}}

Current value-driven cluster context:
{context_json}
"""


def build_multi_model_agentic_forest_prompt(
    context: Dict[str, Any],
    search_config: AgenticFeatureSearchConfig,
) -> str:
    """Construct the proposal prompt for BoW, embedding, and HTR evidence review."""
    context_json = json.dumps(
        context,
        separators=(",", ":"),
        default=_json_default,
    )
    max_proposals = int(
        context.get(
            "max_proposals",
            max(1, int(getattr(search_config, "max_additions_per_iter", 6))),
        )
    )
    concept_inventory_rule = ""
    if context.get("concept_inventory"):
        concept_inventory_rule = (
            "- A separate source-grounded concept_inventory is included. Use it "
            "as a recall checklist for concepts repeatedly represented in BoW, "
            "embedding, or HTR evidence. It is broader than the final feature "
            "set; propose only variables that satisfy the rules below.\n"
        )
    return f"""You are helping convert multi-view sparse bag-of-words, embedding-retrieval, and HTR attention/span evidence into explicit variables for causal inference.

The upstream models are multi-model:
- each configured bag-of-words view has its own vectorizer/model settings;
- each view cross-fits treatment and outcome nuisance models;
- each view computes its own R pseudo-target as (Y - m_hat) / (T - e_hat);
- each view fits a bag-of-words regression model for that pseudo-target;
- feature_importance.views contains the per-view outputs, and
  feature_importance.phrase_consensus summarizes repeated phrase evidence.
- embedding contrasts rank real text chunks and concept phrases by alignment
  with treatment, outcome, and per-view or ensemble R-pseudo-target directions.
- HTR evidence, when present, contributes nuisance predictions to the ensemble
  R signal and highlights attended tokens/spans for treatment, outcome, and
  effect-modifier hypotheses.

Your task is to propose at most {max_proposals} extractable pre-treatment
patient-level candidate variables for a downstream causal forest. Your proposals
are not final causal claims; the workflow will validate complete-document
extraction, fold-honest signal, parsimony, and held-out ITE provenance before
retaining variables.

Rules:
- Propose variables, not raw tokens. Convert token patterns into precise extractable patient-level variables.
- Review concept_inventory first when present so mundane but repeatedly
  represented patient-level fields are not missed.
{concept_inventory_rule}
- Start from the supplied empirical text-model evidence. Do not invent a broad
  hand-built clinical inventory unsupported by BoW, embedding, or HTR evidence
  in this context.
- Review all feature_importance.views. Repeated support across views is stronger,
  but a clinically coherent single-view signal can still be worth proposing.
- Pay special attention to feature_importance.phrase_consensus: it summarizes
  repeated 2-4 token n-gram signals with treatment, outcome,
  confounder-overlap, and pseudo-target evidence.
- When embedding_contrast_evidence is present, use the retrieved real text chunks
  and concept_probe_scores as supporting examples. Do not claim that the vector
  difference itself has a directly decoded meaning.
- Variables supported by both treatment and outcome feature weights should usually be confounders.
- Variables supported by pseudo-target feature weights should usually be effect modifiers.
- A variable may have both roles when justified.
- Use only pre-treatment information.
- Avoid treatment, post-treatment response, toxicity after treatment, survival, and outcome-derived variables.
- For categorical variables, provide 2-8 mutually exclusive categories.
- Do not duplicate current_features.

Return JSON only with this shape:
{{
  "proposals": [
    {{
      "action": "add|none",
      "name": "snake_case_variable_name",
      "type": "categorical|continuous",
      "categories": ["category_a", "category_b"],
      "roles": ["confounder", "effect_modifier"],
      "description": "exact pre-treatment extraction target",
      "rationale": "which BoW features support this variable",
      "expected_signal": "treatment, outcome, or pseudo-target signal expected"
    }}
  ]
}}

Current feature-review context:
{context_json}
"""


def build_multi_model_agentic_consistency_prompt(
    context: Dict[str, Any],
    search_config: AgenticFeatureSearchConfig,
) -> str:
    """Construct the consistency-selection prompt for multi-model candidates."""
    del search_config
    max_selected = int(context.get("max_selected_candidates", 6))
    context_json = json.dumps(
        context,
        separators=(",", ":"),
        default=_json_default,
    )
    return f"""You are selecting stable explicit variables for a causal forest from candidates proposed on separate inner folds.

The outer test fold is not included here. All evidence comes from the current outer-train data only.

Your goals:
- Return the complete, exhaustive keep-list for the next stage. This response replaces the candidate set; any candidate you omit will be discarded.
- Keep every candidate with passes_consistency_gate=true when the gate-passing set fits within the selection limit. Do not omit an obvious or stable candidate on the assumption that another stage retains it automatically.
- Only after including the gate-passing candidates, recover plausible variables that missed the gate but have strong full outer-train evidence, coherent roles, or clear alias-stable support.
- Preserve honest causal inference: use only the supplied candidate summaries and do not invent new variables.
- Temporal eligibility has already been enforced upstream by restricting source inputs to information available before the treatment decision. Do not independently reject a supplied candidate merely because its name or clinical meaning refers to treatment, response, outcome, survival, or toxicity.

Return JSON only with this shape:
{{
  "proposals": [
    {{
      "action": "add|none",
      "name": "existing_candidate_name",
      "type": "categorical|continuous",
      "categories": ["category_a", "category_b"],
      "roles": ["confounder", "effect_modifier"],
      "description": "exact extraction target represented in the supplied evidence",
      "rationale": "why this candidate is stable enough or should be recovered",
      "expected_signal": "treatment, outcome, or pseudo-target signal expected"
    }}
  ]
}}

Rules:
- Choose only names listed in candidate_summaries.
- Treat passes_consistency_gate=true as a keep decision, not merely a preference. Include every such candidate as an add proposal unless the gate-passing set alone exceeds the limit or the candidate is an exact duplicate that cannot be represented separately.
- If gate-passing candidates exceed the limit, keep the highest-support candidates first and return action=none for each omitted gate-passing candidate with a specific rationale.
- Do not spend selection capacity on below-threshold recovery candidates until all gate-passing candidates have been included.
- You may then recover a below-threshold candidate only if it has strong full_outer_train support or a clear explanation for fold instability.
- Do not apply an additional temporal or outcome-semantic exclusion; the supplied candidate evidence is already restricted to the permitted decision-time information.
- Keep roles tied to evidence: treatment+outcome support implies confounder; pseudo-target support implies effect_modifier; both signals may justify both roles.
- Return at most {max_selected} add proposals. Your add proposals are the entire downstream keep-list, not a list of changes or exceptions.

Current consistency-selection context:
{context_json}
"""


def build_multi_model_agentic_extracted_feature_review_prompt(
    context: Dict[str, Any],
    search_config: AgenticFeatureSearchConfig,
) -> str:
    """Construct the post-extraction diagnostic review prompt."""
    context_json = json.dumps(
        context,
        separators=(",", ":"),
        default=_json_default,
    )
    max_proposals = int(
        context.get(
            "max_proposals",
            max(1, int(getattr(search_config, "max_additions_per_iter", 6))),
        )
    )
    max_removals = int(getattr(search_config, "max_removals_per_iter", 3))
    return f"""You are reviewing extracted explicit variables before a downstream causal forest is fit.

The outer test fold is not included here. All diagnostics come from cross-fitted models on the current outer-training fold only.

The selected candidate variables have already been extracted from clinical text. Simple nuisance and R/pseudo-target models were trained on those extracted values and compared with the original multi-view BoW, embedding-contrast, and HTR evidence. Your task is to propose evidence-supported revisions when the extracted variables do not preserve the confounder or effect-modifier signal seen upstream.

Temporal eligibility is enforced upstream by restricting source inputs to information available before the treatment decision. Do not independently exclude a supplied variable because its name or clinical meaning refers to treatment, response, outcome, survival, or toxicity.

Return JSON only with this shape:
{{
  "proposals": [
    {{
      "action": "add|remove|update_role|none",
      "name": "snake_case_variable_name",
      "type": "categorical|continuous",
      "categories": ["category_a", "category_b"],
      "roles": ["confounder", "effect_modifier"],
      "description": "exact extraction target represented in the supplied evidence",
      "rationale": "why this change addresses the diagnostic failure",
      "expected_signal": "treatment, outcome, or pseudo-target signal expected"
    }}
  ]
}}

Rules:
- Return at most {max_proposals} add proposals and at most {max_removals} remove proposals.
- Use update_role when a feature was extracted correctly but assigned to the wrong causal role.
- Use remove when an agent-added feature is too sparse, constant, redundant, or unsupported by diagnostics.
- Do not remove required_features; propose role updates for them only when diagnostics justify it.
- Add a replacement only when the BoW, embedding, uplift, or HTR evidence points to a clearer extractable patient-level variable.
- Inspect htr_attention_evidence snippets and attended token spans directly when deciding which missing or broader variable could preserve HTR nuisance, effect, or pair-uplift signal.
- Do not add variables from general clinical intuition alone; tie revisions to the supplied upstream evidence or extracted-feature diagnostics.
- For categorical variables, provide 2-8 mutually exclusive categories.
- Do not apply an additional temporal or outcome-semantic exclusion; the supplied evidence has already been restricted to the permitted decision-time information.
- Use "none" if the current extracted feature set is the best defensible set despite the diagnostic gap.

Current extracted-feature review context:
{context_json}
"""


def build_broad_agent_prompt(
    context: Dict[str, Any],
    search_config: AgenticFeatureSearchConfig,
) -> str:
    """Construct a high-recall initial proposal prompt for broad-screen mode."""
    context_json = json.dumps(context, indent=2, default=_json_default)
    candidate_count = int(context.get("broad_candidate_count", search_config.broad_candidate_count))
    return f"""You are helping design a high-recall baseline variable inventory for causal inference.

Propose a broad list of variables that are plausibly extractable from the text and could act as pre-treatment confounders or treatment effect modifiers.
This is only the first-pass extractable inventory. The next stage will statistically screen candidates and then an agent will adaptively select from the extracted shortlist or define new features if needed, so prioritize recall and precise extraction definitions over narrow confidence.

Return JSON only with this shape:
{{
  "proposals": [
    {{
      "action": "add",
      "name": "snake_case_variable_name",
      "type": "categorical|continuous",
      "categories": ["category_a", "category_b"],
      "roles": ["confounder", "effect_modifier"],
      "description": "exact pre-treatment extraction target",
      "rationale": "why this may help",
      "expected_signal": "treatment, outcome, or tau signal expected"
    }}
  ]
}}

Limits:
- Return up to {candidate_count} add proposals.
- Do not return remove, update_role, or none actions in this mode.
- Do not duplicate any feature listed in required_features.
- Each variable must be measurable before or at treatment initiation.
- Do not propose treatment, post-treatment response, survival, toxicity, or outcome-derived variables.
- For categorical variables, provide 2-8 mutually exclusive categories.
- Use distinct variable names; avoid near-duplicate aliases for the same concept.

Current train-fold context:
{context_json}
"""


def build_broad_selection_agent_prompt(
    context: Dict[str, Any],
    search_config: AgenticFeatureSearchConfig,
) -> str:
    """Construct the adaptive selection prompt after broad-screen extraction."""
    context_json = json.dumps(context, indent=2, default=_json_default)
    return f"""You are helping adaptively select explicit variables for a causal forest.

Broad-screen extraction has already produced the ranked available_extracted_features in the context. You may select from that shortlist by returning an add proposal with the existing feature name, or you may define a new pre-treatment extractable feature when the current metrics and feedback suggest the extracted shortlist is missing an important concept.

Return JSON only with this shape:
{{
  "proposals": [
    {{
      "action": "add|remove|update_role|none",
      "name": "snake_case_variable_name",
      "type": "categorical|continuous",
      "categories": ["category_a", "category_b"],
      "roles": ["confounder", "effect_modifier"],
      "description": "exact extraction target",
      "rationale": "why this may help",
      "expected_signal": "treatment, outcome, or tau signal expected"
    }}
  ]
}}

Limits:
- At most {search_config.max_additions_per_iter} add proposals.
- At most {search_config.max_removals_per_iter} remove proposals.
- To select an available_extracted_features item, use its exact name; its already-extracted contract will be used.
- For a new add proposal not in available_extracted_features, provide type, roles, description, and categories when categorical.
- Do not duplicate current_features or required_features.
- Use "none" if no defensible add, remove, or role update is available.
- Review iteration_feedback and recent_decisions before proposing.

Current broad-screen selection context:
{context_json}
"""


def build_agent_repair_prompt(issues: Sequence[str]) -> str:
    """Construct a follow-up prompt asking the agent to repair proposal JSON."""
    issue_lines = "\n".join(f"- {issue}" for issue in issues)
    return f"""The previous response could not be used because it failed these schema checks:
{issue_lines}

Return corrected JSON only. Use this exact top-level shape:
{{
  "proposals": [
    {{
      "action": "add|remove|update_role|none",
      "name": "snake_case_variable_name",
      "type": "categorical|continuous",
      "categories": ["category_a", "category_b"],
      "roles": ["confounder", "effect_modifier"],
      "description": "exact extraction target",
      "rationale": "why this may help",
      "expected_signal": "treatment, outcome, or tau signal expected"
    }}
  ]
}}

Repair the same candidate concepts when possible. Do not add prose, markdown,
comments, or code fences. For every add proposal, include a non-empty roles
array containing only "confounder" and/or "effect_modifier".
"""


def build_consensus_disambiguation_repair_prompt(issues: Sequence[str]) -> str:
    """Construct a follow-up prompt asking the agent to repair group JSON."""
    issue_lines = "\n".join(f"- {issue}" for issue in issues)
    return f"""The previous response could not be used because it failed these schema checks:
{issue_lines}

Return corrected JSON only. Use this exact top-level shape:
{{
  "groups": [
    {{
      "canonical_name": "one_existing_snake_case_member_name",
      "member_names": ["existing_proposal_name_a", "existing_proposal_name_b"],
      "member_folds": [1, 2],
      "type": "categorical|continuous",
      "categories": ["category_a", "category_b"],
      "description": "exact shared extraction target",
      "rationale": "brief reason these names are aliases"
    }}
  ],
  "unmerged": [
    {{
      "name": "existing_proposal_name",
      "reason": "why this was not equivalent to another proposal"
    }}
  ]
}}

Do not add prose, markdown, comments, or code fences.
"""


def build_value_harmonization_repair_prompt(issues: Sequence[str]) -> str:
    """Construct a follow-up prompt asking the agent to repair value-contract JSON."""
    issue_lines = "\n".join(f"- {issue}" for issue in issues)
    return f"""The previous response could not be used because it failed these schema checks:
{issue_lines}

Return corrected JSON only. Use this exact top-level shape:
{{
  "features": [
    {{
      "name": "existing_variable_name",
      "type": "categorical|continuous",
      "categories": ["canonical_category_a", "canonical_category_b"],
      "description": "exact extraction target and value policy",
      "value_aliases": {{
        "canonical_category_a": ["alias_a", "alias_b"]
      }},
      "missing_values": ["unknown", "not_reported"],
      "rationale": "brief reason for the value contract"
    }}
  ]
}}

Do not add prose, markdown, comments, or code fences.
"""


def build_concept_inventory_repair_prompt(issues: Sequence[str]) -> str:
    """Construct a follow-up prompt asking the agent to repair concept inventory JSON."""
    issue_lines = "\n".join(f"- {issue}" for issue in issues)
    return f"""The previous response could not be used because it failed these schema checks:
{issue_lines}

Return corrected JSON only. Use this exact top-level shape:
{{
  "concepts": [
    {{
      "name": "snake_case_concept_name",
      "label": "short human-readable label",
      "value_kind": "binary|categorical|continuous|ordinal|text|unknown",
      "source_families": ["bow", "embedding_contrast", "htr"],
      "source_overlap": 2,
      "supporting_phrases": ["short phrase or token evidence"],
      "example_values_or_phrases": ["example values or note phrases"],
      "extractability": "high|medium|low",
      "cluster_ids": ["cluster_001"],
      "notes": "brief source-grounded explanation"
    }}
  ],
  "omitted_uncertain": ["optional short labels for weak one-source concepts"],
  "rejected_clusters": [
    {{
      "cluster_id": "cluster_009",
      "reason": "brief reason this cluster was not a patient-level concept"
    }}
  ]
}}

Do not add prose, markdown, comments, or code fences.
"""


def build_parsimony_factor_repair_prompt(issues: Sequence[str]) -> str:
    """Construct a repair request for cluster-to-factor response JSON."""
    issue_lines = "\n".join(f"- {issue}" for issue in issues)
    return f"""The previous response could not be used because it failed these schema checks:
{issue_lines}

Return corrected JSON only with this exact top-level shape:
{{
  "cluster_id": "the_supplied_cluster_id",
  "decision": "replace_cluster|retain_cluster",
  "replaces": ["replaceable_cluster_member"],
  "factors": [
    {{
      "name": "snake_case_factor_name",
      "inference_kind": "implicit|direct",
      "type": "categorical|continuous",
      "categories": ["category_a", "category_b"],
      "unit": "measurement unit for continuous factors or null",
      "roles": ["confounder", "effect_modifier"],
      "description": "precise patient-level factor",
      "supporting_indicators": ["specific supporting evidence"],
      "contrary_indicators": ["specific contrary evidence"],
      "minimum_evidence": "minimum evidence for a non-null value",
      "null_policy": "when to return null",
      "rationale": "why the factor represents the replaced members"
    }}
  ],
  "rationale": "brief cluster-level decision rationale"
}}

Do not add prose, markdown, comments, or code fences.
"""


def agent_response_schema_issues(
    proposals: Sequence[Any],
    context: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Return schema-level issues that are worth asking the LLM to repair."""
    issues: List[str] = []
    available_names = _context_available_extracted_names(context)
    allow_missing_roles = _context_allows_missing_roles(context)
    for idx, raw in enumerate(proposals, start=1):
        if not isinstance(raw, dict):
            issues.append(f"proposal {idx}: expected an object, got {type(raw).__name__}")
            continue

        label = _proposal_schema_label(idx, raw)
        action_raw = raw.get("action")
        action = str(action_raw).strip().lower() if action_raw is not None else ""
        if not action:
            issues.append(f"{label}: missing action")
            continue
        if action not in VALID_ACTIONS:
            issues.append(
                f"{label}: invalid action {action_raw!r}; expected one of {sorted(VALID_ACTIONS)}"
            )
            continue
        if action == "none":
            continue

        if _missing_or_empty(raw.get("name")):
            issues.append(f"{label}: missing name")

        if action == "add":
            if _normalize_feature_name(raw.get("name", "")) in available_names:
                continue
            proposal_type = raw.get("type")
            if _missing_or_empty(proposal_type):
                issues.append(f"{label}: missing type")
            elif proposal_type not in VALID_TYPES:
                issues.append(
                    f"{label}: invalid type {proposal_type!r}; expected one of {sorted(VALID_TYPES)}"
                )

            roles_issue = _roles_schema_issue(
                raw.get("roles"),
                allow_missing=allow_missing_roles,
            )
            if roles_issue is not None:
                issues.append(f"{label}: {roles_issue}")

            if proposal_type == "categorical" and not raw.get("categories"):
                issues.append(f"{label}: missing categories for categorical proposal")
            if _missing_or_empty(raw.get("description")):
                issues.append(f"{label}: missing description")
        elif action == "update_role":
            roles_issue = _roles_schema_issue(
                raw.get("roles"),
                allow_missing=allow_missing_roles,
            )
            if roles_issue is not None:
                issues.append(f"{label}: {roles_issue}")

    return issues


def consensus_disambiguation_response_issues(response: Any) -> List[str]:
    """Return minimal schema issues for consensus-disambiguation responses."""
    issues: List[str] = []
    if not isinstance(response, dict):
        return [f"expected a JSON object, got {type(response).__name__}"]
    groups = response.get("groups")
    if groups is None:
        issues.append("missing groups list")
    elif not isinstance(groups, list):
        issues.append("groups must be a list")
    unmerged = response.get("unmerged", [])
    if unmerged is not None and not isinstance(unmerged, list):
        issues.append("unmerged must be a list when provided")
    return issues


def value_harmonization_response_issues(
    response: Any,
    context: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Return schema issues for value-harmonization responses."""
    if not isinstance(response, dict):
        return [f"expected a JSON object, got {type(response).__name__}"]
    features = response.get("features")
    if not isinstance(features, list):
        return ["missing features list"]

    expected_names = set()
    if isinstance(context, dict):
        for item in context.get("selected_features", []):
            if isinstance(item, dict) and item.get("name"):
                expected_names.add(_normalize_feature_name(item["name"]))

    issues: List[str] = []
    seen_names = set()
    for idx, item in enumerate(features, start=1):
        if not isinstance(item, dict):
            issues.append(f"feature {idx}: expected an object")
            continue
        name = _normalize_feature_name(item.get("name", ""))
        if not name:
            issues.append(f"feature {idx}: missing name")
            continue
        if expected_names and name not in expected_names:
            issues.append(f"feature {idx} ({name}): name was not in selected_features")
        if name in seen_names:
            issues.append(f"feature {idx} ({name}): duplicate feature")
        seen_names.add(name)

        feature_type = str(item.get("type", "")).strip().lower()
        if feature_type not in VALID_TYPES:
            issues.append(
                f"feature {idx} ({name}): invalid type {item.get('type')!r}; "
                f"expected one of {sorted(VALID_TYPES)}"
            )
            continue

        categories = item.get("categories")
        if feature_type == "categorical":
            if not isinstance(categories, list) or not categories:
                issues.append(f"feature {idx} ({name}): categorical feature needs categories")
            elif len(categories) > 8:
                issues.append(f"feature {idx} ({name}): too many categories")
        elif categories not in (None, []):
            issues.append(f"feature {idx} ({name}): continuous feature must not have categories")

        if _missing_or_empty(item.get("description")):
            issues.append(f"feature {idx} ({name}): missing description")

    missing = sorted(expected_names - seen_names)
    if missing:
        issues.append(f"missing selected feature(s): {missing}")
    return issues


def concept_inventory_response_issues(
    response: Any,
    context: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Return schema issues for source-grounded concept inventory responses."""
    if not isinstance(response, dict):
        return [f"expected a JSON object, got {type(response).__name__}"]
    concepts = response.get("concepts")
    if not isinstance(concepts, list):
        return ["missing concepts list"]

    max_concepts = None
    if isinstance(context, dict) and context.get("max_concepts") is not None:
        try:
            max_concepts = int(context.get("max_concepts"))
        except (TypeError, ValueError):
            max_concepts = None
    prompt_version = context.get("prompt_version") if isinstance(context, dict) else None
    is_cluster_labeling = prompt_version in {
        "multi_model_agentic_cluster_labeling_v1",
        "multi_model_agentic_cluster_labeling_v2",
    }
    known_cluster_ids = set()
    if isinstance(context, dict):
        for cluster in context.get("clusters") or []:
            if isinstance(cluster, dict) and cluster.get("cluster_id"):
                known_cluster_ids.add(str(cluster.get("cluster_id")))

    issues: List[str] = []
    seen_names = set()
    for idx, item in enumerate(concepts, start=1):
        if not isinstance(item, dict):
            issues.append(f"concept {idx}: expected an object")
            continue
        name = _normalize_feature_name(item.get("name", ""))
        if not name:
            issues.append(f"concept {idx}: missing name")
            continue
        if name in seen_names:
            issues.append(f"concept {idx} ({name}): duplicate concept name")
        seen_names.add(name)
        source_families = item.get("source_families", [])
        if source_families is not None and not isinstance(source_families, list):
            issues.append(f"concept {idx} ({name}): source_families must be a list")
        if item.get("supporting_phrases") is not None and not isinstance(
            item.get("supporting_phrases"),
            list,
        ):
            issues.append(f"concept {idx} ({name}): supporting_phrases must be a list")
        cluster_ids = item.get("cluster_ids")
        if cluster_ids is not None and not isinstance(cluster_ids, list):
            issues.append(f"concept {idx} ({name}): cluster_ids must be a list")
        elif cluster_ids is not None:
            for cluster_id in cluster_ids:
                cluster_id_text = str(cluster_id)
                if known_cluster_ids and cluster_id_text not in known_cluster_ids:
                    issues.append(f"concept {idx} ({name}): unknown cluster_id {cluster_id_text!r}")
        if is_cluster_labeling and not cluster_ids:
            issues.append(f"concept {idx} ({name}): cluster_ids required for cluster labeling")

    if max_concepts is not None and len(concepts) > max_concepts * 2:
        issues.append(
            f"concepts list has {len(concepts)} entries; expected at most about {max_concepts}"
        )
    omitted = response.get("omitted_uncertain", [])
    if omitted is not None and not isinstance(omitted, list):
        issues.append("omitted_uncertain must be a list when provided")
    rejected = response.get("rejected_clusters", [])
    if rejected is not None and not isinstance(rejected, list):
        issues.append("rejected_clusters must be a list when provided")
    elif isinstance(rejected, list):
        for idx, item in enumerate(rejected, start=1):
            if not isinstance(item, dict):
                issues.append(f"rejected_clusters {idx}: expected an object")
                continue
            cluster_id = str(item.get("cluster_id") or "")
            if not cluster_id:
                issues.append(f"rejected_clusters {idx}: missing cluster_id")
            elif known_cluster_ids and cluster_id not in known_cluster_ids:
                issues.append(f"rejected_clusters {idx}: unknown cluster_id {cluster_id!r}")
            if _missing_or_empty(item.get("reason")):
                issues.append(f"rejected_clusters {idx} ({cluster_id}): missing reason")
    return issues


def parsimony_factor_response_issues(
    response: Any,
    context: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Return schema issues for value-cluster factorization responses."""
    if not isinstance(response, dict):
        return [f"expected a JSON object, got {type(response).__name__}"]

    context = context if isinstance(context, dict) else {}
    expected_cluster_id = str(context.get("cluster_id") or "")
    cluster_id = str(response.get("cluster_id") or "")
    issues: List[str] = []
    if not cluster_id:
        issues.append("missing cluster_id")
    elif expected_cluster_id and cluster_id != expected_cluster_id:
        issues.append(f"cluster_id {cluster_id!r} does not match supplied {expected_cluster_id!r}")

    decision = str(response.get("decision") or "").strip().lower()
    if decision not in {"replace_cluster", "retain_cluster"}:
        issues.append("decision must be replace_cluster or retain_cluster")
        return issues

    factors = response.get("factors", [])
    replaces = response.get("replaces", [])
    if not isinstance(factors, list):
        issues.append("factors must be a list")
        factors = []
    if not isinstance(replaces, list):
        issues.append("replaces must be a list")
        replaces = []

    if decision == "retain_cluster":
        if factors:
            issues.append("retain_cluster must not include factors")
        if replaces:
            issues.append("retain_cluster must not include replaces")
        if _missing_or_empty(response.get("rationale")):
            issues.append("retain_cluster needs a rationale")
        return issues

    max_factors = int(context.get("max_factors", 2) or 2)
    if not 1 <= len(factors) <= max_factors:
        issues.append(f"replace_cluster needs 1-{max_factors} factors")

    replaceable = {
        _normalize_feature_name(name)
        for name in context.get("replaceable_members", []) or []
        if _normalize_feature_name(name)
    }
    protected = {
        _normalize_feature_name(name)
        for name in context.get("protected_members", []) or []
        if _normalize_feature_name(name)
    }
    normalized_replaces = [_normalize_feature_name(name) for name in replaces]
    if len(set(normalized_replaces)) < 2:
        issues.append("replace_cluster must replace at least two distinct members")
    for name in normalized_replaces:
        if name not in replaceable:
            issues.append(f"replacement member {name!r} was not replaceable")
        if name in protected:
            issues.append(f"replacement member {name!r} is protected")
    if len(factors) >= len(set(normalized_replaces)) and factors:
        issues.append("factor count must be smaller than replaced-member count")

    expected_roles = {
        str(role)
        for role in context.get("required_role_union", []) or []
        if str(role) in VALID_ROLES
    }
    factor_roles: set = set()
    seen_names: set = set()
    for idx, factor in enumerate(factors, start=1):
        if not isinstance(factor, dict):
            issues.append(f"factor {idx}: expected an object")
            continue
        name = _normalize_feature_name(factor.get("name", ""))
        if not name:
            issues.append(f"factor {idx}: missing name")
        elif name in seen_names:
            issues.append(f"factor {idx} ({name}): duplicate name")
        seen_names.add(name)

        inference_kind = str(factor.get("inference_kind") or "").strip().lower()
        if inference_kind not in {"implicit", "direct"}:
            issues.append(f"factor {idx} ({name}): invalid inference_kind")
        feature_type = str(factor.get("type") or "").strip().lower()
        if feature_type not in VALID_TYPES:
            issues.append(f"factor {idx} ({name}): invalid type")
        categories = factor.get("categories")
        if feature_type == "categorical":
            if not isinstance(categories, list) or not 2 <= len(categories) <= 8:
                issues.append(f"factor {idx} ({name}): categorical factor needs 2-8 categories")
        elif categories not in (None, []):
            issues.append(f"factor {idx} ({name}): continuous factor cannot have categories")
        if feature_type == "continuous" and _missing_or_empty(factor.get("unit")):
            issues.append(f"factor {idx} ({name}): continuous factor needs a unit")
        if inference_kind == "implicit" and feature_type == "continuous":
            issues.append(f"factor {idx} ({name}): implicit factors must be categorical")

        roles = factor.get("roles")
        if not isinstance(roles, list) or not roles:
            issues.append(f"factor {idx} ({name}): missing roles")
        else:
            invalid_roles = {str(role) for role in roles} - VALID_ROLES
            if invalid_roles:
                issues.append(f"factor {idx} ({name}): invalid roles {sorted(invalid_roles)}")
            factor_roles.update(str(role) for role in roles if str(role) in VALID_ROLES)

        for field_name in [
            "description",
            "minimum_evidence",
            "null_policy",
            "rationale",
        ]:
            if _missing_or_empty(factor.get(field_name)):
                issues.append(f"factor {idx} ({name}): missing {field_name}")
        for field_name in ["supporting_indicators", "contrary_indicators"]:
            value = factor.get(field_name)
            if not isinstance(value, list):
                issues.append(f"factor {idx} ({name}): {field_name} must be a list")
        if not factor.get("supporting_indicators"):
            issues.append(f"factor {idx} ({name}): supporting_indicators cannot be empty")

    if expected_roles and factor_roles != expected_roles:
        issues.append(
            "factor roles must collectively equal required_role_union "
            f"{sorted(expected_roles)}; got {sorted(factor_roles)}"
        )
    if _missing_or_empty(response.get("rationale")):
        issues.append("replace_cluster needs a rationale")
    return issues


def _context_available_extracted_names(
    context: Optional[Dict[str, Any]],
) -> set:
    if not isinstance(context, dict):
        return set()
    if context.get("search_mode") != "broad_screen":
        return set()
    if context.get("broad_screen_stage") != "selection":
        return set()
    available = context.get("available_extracted_features", [])
    if not isinstance(available, list):
        return set()
    names = set()
    for item in available:
        if isinstance(item, dict) and item.get("name"):
            names.add(_normalize_feature_name(item["name"]))
    return names


def _context_allows_missing_roles(context: Optional[Dict[str, Any]]) -> bool:
    return bool(
        isinstance(context, dict)
        and context.get("prompt_version") == "agentic_attention_variable_forest_v1"
    )


def _roles_schema_issue(roles: Any, allow_missing: bool = False) -> Optional[str]:
    if roles is None or roles == []:
        if allow_missing:
            return None
        return "missing roles"
    role_values = [roles] if isinstance(roles, str) else roles
    if not isinstance(role_values, list):
        return "roles must be a string or list"
    normalized = [str(role).strip() for role in role_values if str(role).strip()]
    if not normalized:
        return "missing roles"
    invalid = sorted(set(normalized) - VALID_ROLES)
    if invalid:
        return f"invalid roles {invalid}; expected one or both of {sorted(VALID_ROLES)}"
    return None


def _proposal_schema_label(idx: int, raw: Dict[str, Any]) -> str:
    name = raw.get("name")
    if _missing_or_empty(name):
        return f"proposal {idx}"
    return f"proposal {idx} ({name})"


def _missing_or_empty(value: Any) -> bool:
    if value is None:
        return True
    return isinstance(value, str) and not value.strip()


def parse_agent_response(response: str) -> List[Dict[str, Any]]:
    """Parse JSON proposals from an LLM response."""
    parsed = parse_agent_json_object_or_list(response)
    if isinstance(parsed, list):
        return parsed
    proposals = parsed.get("proposals", [])
    if not isinstance(proposals, list):
        raise ValueError("Agent response JSON must contain a proposals list")
    return proposals


def parse_agent_json_object(response: str) -> Dict[str, Any]:
    """Parse a top-level JSON object from an LLM response."""
    parsed = parse_agent_json_object_or_list(response)
    if not isinstance(parsed, dict):
        raise ValueError(f"Agent response JSON must be an object, got {type(parsed).__name__}")
    return parsed


def parse_agent_json_object_or_list(response: str) -> Any:
    """Parse a JSON object or list from an LLM response."""
    response = strip_reasoning_trace(response)
    match = re.search(r"\{.*\}", response, re.DOTALL)
    json_str = match.group(0) if match else response
    return json.loads(json_str)


def _chat_completion_trace(
    response: Any,
    choice: Any,
    message: Any,
    content: str,
) -> Dict[str, Any]:
    """Build a JSON-serializable trace of the proposal agent response."""
    reasoning_content = _message_field(message, "reasoning_content")
    reasoning = _message_field(message, "reasoning")
    trace = {
        "raw_content": content,
        "finish_reason": getattr(choice, "finish_reason", None),
        "model": getattr(response, "model", None),
        "response_id": getattr(response, "id", None),
        "created": getattr(response, "created", None),
        "usage": _to_jsonable(getattr(response, "usage", None)),
    }
    if reasoning_content is not None:
        trace["reasoning_content"] = reasoning_content
    if reasoning is not None:
        trace["reasoning"] = reasoning
    return {key: value for key, value in trace.items() if value is not None}


def _trace_with_repair_attempts(
    trace: Dict[str, Any],
    attempts: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    if len(attempts) <= 1:
        return trace
    enriched = dict(trace)
    enriched["repair_attempts"] = list(attempts)
    return enriched


def _message_field(message: Any, key: str) -> Any:
    value = getattr(message, key, None)
    if value is not None:
        return _to_jsonable(value)

    extra = getattr(message, "model_extra", None)
    if isinstance(extra, dict) and extra.get(key) is not None:
        return _to_jsonable(extra[key])

    dumped = _model_dump(message)
    if isinstance(dumped, dict) and dumped.get(key) is not None:
        return _to_jsonable(dumped[key])

    return None


def _get_agent_response_trace(agent: Any) -> Dict[str, Any]:
    trace = getattr(agent, "last_response_trace", None)
    if trace is not None:
        return _to_jsonable(trace)

    raw_response = getattr(agent, "last_raw_response", None)
    if raw_response is not None:
        return {"raw_content": str(raw_response)}

    return {"available": False}


def _model_dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json")
        except TypeError:
            return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return None


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]

    dumped = _model_dump(value)
    if dumped is not None and dumped is not value:
        return _to_jsonable(dumped)

    return str(value)


def validate_agentic_proposals(
    raw_proposals: Sequence[Dict[str, Any]],
    current_specs: List[ExplicitFeatureSpec],
    search_config: AgenticFeatureSearchConfig,
    allow_removals: bool,
    max_additions: Optional[int] = None,
    allow_duplicate_additions: bool = False,
) -> Tuple[List[AgenticFeatureProposal], List[Dict[str, Any]]]:
    """Validate raw LLM proposals against schema, role, and leakage guards."""
    current_names = {spec.name for spec in current_specs}
    valid: List[AgenticFeatureProposal] = []
    rejected = []
    additions = 0
    removals = 0
    max_additions = (
        search_config.max_additions_per_iter if max_additions is None else int(max_additions)
    )

    for raw in raw_proposals:
        try:
            proposal = _coerce_proposal(raw)
            reason = _proposal_rejection_reason(proposal, current_names, allow_removals)
            if reason is None and proposal.action == "add":
                additions += 1
                if additions > max_additions:
                    reason = "too_many_additions"
            if reason is None and proposal.action == "remove":
                removals += 1
                if removals > search_config.max_removals_per_iter:
                    reason = "too_many_removals"
            if reason is None and proposal.action != "none":
                valid.append(proposal)
                if proposal.action == "add" and not allow_duplicate_additions:
                    current_names.add(proposal.name)
            elif reason is not None:
                rejected.append({"proposal": raw, "reason": reason})
        except Exception as exc:
            rejected.append({"proposal": raw, "reason": str(exc)})

    return valid, rejected


def apply_proposals(
    current_specs: List[ExplicitFeatureSpec],
    proposals: Sequence[AgenticFeatureProposal],
) -> List[ExplicitFeatureSpec]:
    """Apply one or more validated proposals to a feature spec list."""
    specs = list(current_specs)
    for proposal in proposals:
        if proposal.action == "add":
            specs.append(
                ExplicitFeatureSpec(
                    name=proposal.name,
                    type=proposal.type or "continuous",
                    categories=proposal.categories,
                    description=proposal.description,
                    roles=proposal.roles,
                )
            )
        elif proposal.action == "remove":
            specs = [spec for spec in specs if spec.name != proposal.name]
        elif proposal.action == "update_role":
            updated = []
            for spec in specs:
                if spec.name == proposal.name:
                    updated.append(
                        ExplicitFeatureSpec(
                            name=spec.name,
                            type=spec.type,
                            categories=spec.categories,
                            description=spec.description,
                            value_aliases=getattr(spec, "value_aliases", None),
                            roles=proposal.roles,
                        )
                    )
                else:
                    updated.append(spec)
            specs = updated
    return specs


def apply_agentic_value_harmonization(
    *,
    specs: Sequence[ExplicitFeatureSpec],
    response: Any,
) -> Tuple[List[ExplicitFeatureSpec], List[Dict[str, Any]]]:
    """Apply an agent-reviewed value contract to already-selected specs."""
    if not isinstance(response, dict):
        return list(specs), [{"error": "response_not_object"}]
    raw_features = response.get("features")
    if not isinstance(raw_features, list):
        return list(specs), [{"error": "missing_features_list"}]

    by_name = {spec.name: spec for spec in specs}
    harmonized = {spec.name: spec for spec in specs}
    applied: List[Dict[str, Any]] = []
    for item in raw_features:
        if not isinstance(item, dict):
            applied.append({"ignored": item, "reason": "feature_not_object"})
            continue
        name = _normalize_feature_name(item.get("name", ""))
        spec = by_name.get(name)
        if spec is None:
            applied.append({"name": name, "reason": "unknown_feature"})
            continue

        feature_type = str(item.get("type") or spec.type).strip().lower()
        if feature_type not in VALID_TYPES:
            feature_type = spec.type

        description = str(item.get("description") or spec.description or "").strip()
        if feature_type == "categorical":
            categories = _canonical_value_categories(item.get("categories") or spec.categories)
            if not categories:
                applied.append({"name": name, "reason": "empty_categorical_categories"})
                continue
            value_aliases = _canonical_value_aliases(
                item.get("value_aliases"),
                categories,
            )
            description = _append_value_policy(
                description,
                categories=categories,
                value_aliases=value_aliases,
                continuous=False,
            )
        else:
            categories = None
            value_aliases = None
            description = _append_value_policy(
                description,
                categories=None,
                value_aliases=None,
                continuous=True,
            )

        new_spec = ExplicitFeatureSpec(
            name=spec.name,
            type=feature_type,
            categories=categories,
            description=description or spec.description,
            value_aliases=value_aliases,
            roles=spec.roles,
        )
        harmonized[spec.name] = new_spec
        applied.append(
            {
                "name": spec.name,
                "from": _spec_to_dict(spec),
                "to": _spec_to_dict(new_spec),
                "rationale": item.get("rationale"),
            }
        )

    return _dedupe_agentic_specs(list(harmonized.values())), applied


def apply_agentic_alias_resolution(
    *,
    proposals: Sequence[AgenticFeatureProposal],
    known_specs: Sequence[ExplicitFeatureSpec],
    response: Any,
) -> Tuple[List[AgenticFeatureProposal], List[Dict[str, str]]]:
    """Apply agent-reviewed aliases to add proposals, reusing known canonical specs."""
    if not isinstance(response, dict):
        return list(proposals), []

    proposal_by_name = {
        proposal.name: proposal for proposal in proposals if proposal.action == "add"
    }
    known_by_name = {spec.name: spec for spec in known_specs}
    allowed_names = set(proposal_by_name) | set(known_by_name)
    alias_to_canonical: Dict[str, str] = {}
    applied_aliases: List[Dict[str, str]] = []

    groups = response.get("groups") or []
    if not isinstance(groups, list):
        return list(proposals), []

    for group in groups:
        if not isinstance(group, dict):
            continue
        canonical = _normalize_feature_name(group.get("canonical_name", ""))
        raw_members = group.get("member_names") or []
        if not isinstance(raw_members, list):
            continue
        members = [
            _normalize_feature_name(member)
            for member in raw_members
            if _normalize_feature_name(member) in allowed_names
        ]
        if canonical not in allowed_names:
            continue
        if canonical not in members and canonical not in known_by_name:
            continue
        if len(set([canonical, *members])) < 2:
            continue
        for member in members:
            if member in proposal_by_name and member != canonical:
                alias_to_canonical[member] = canonical
                applied_aliases.append(
                    {
                        "from": member,
                        "to": canonical,
                        "rationale": str(group.get("rationale") or ""),
                    }
                )

    if not alias_to_canonical:
        return list(proposals), []

    rewritten: List[AgenticFeatureProposal] = []
    emitted_add_names: set = set()
    for proposal in proposals:
        if proposal.action != "add":
            rewritten.append(proposal)
            continue
        target_name = alias_to_canonical.get(proposal.name, proposal.name)
        known_spec = known_by_name.get(target_name)
        retargeted = _retarget_proposal(
            proposal=proposal,
            target_name=target_name,
            known_spec=known_spec,
        )
        if target_name in emitted_add_names:
            for index, existing in enumerate(rewritten):
                if existing.action == "add" and existing.name == target_name:
                    rewritten[index] = _merge_proposals(existing, retargeted)
                    break
            continue
        rewritten.append(retargeted)
        emitted_add_names.add(target_name)

    return rewritten, applied_aliases


def _proposal_to_spec(proposal: AgenticFeatureProposal) -> ExplicitFeatureSpec:
    return ExplicitFeatureSpec(
        name=proposal.name,
        type=proposal.type or "continuous",
        categories=proposal.categories,
        description=proposal.description,
        roles=proposal.roles,
    )


def _proposal_from_spec(
    spec: ExplicitFeatureSpec,
    source: Optional[AgenticFeatureProposal] = None,
) -> AgenticFeatureProposal:
    return AgenticFeatureProposal(
        action="add",
        name=spec.name,
        type=spec.type,
        categories=spec.categories,
        description=spec.description,
        roles=list(spec.roles or []),
        rationale=None if source is None else source.rationale,
        expected_signal=None if source is None else source.expected_signal,
    )


def _retarget_proposal(
    *,
    proposal: AgenticFeatureProposal,
    target_name: str,
    known_spec: Optional[ExplicitFeatureSpec],
) -> AgenticFeatureProposal:
    if known_spec is None:
        return AgenticFeatureProposal(
            action=proposal.action,
            name=target_name,
            type=proposal.type,
            categories=proposal.categories,
            description=proposal.description,
            roles=proposal.roles,
            rationale=proposal.rationale,
            expected_signal=proposal.expected_signal,
        )

    categories = _merge_ordered_values(known_spec.categories, proposal.categories) or None
    feature_type = (
        "categorical"
        if known_spec.type == "categorical" or proposal.type == "categorical" or categories
        else known_spec.type
    )
    return AgenticFeatureProposal(
        action=proposal.action,
        name=target_name,
        type=feature_type,
        categories=categories,
        description=_merge_text_values(known_spec.description, proposal.description),
        roles=_merge_ordered_values(known_spec.roles, proposal.roles),
        rationale=proposal.rationale,
        expected_signal=proposal.expected_signal,
    )


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


def _canonical_value_categories(value: Any) -> Optional[List[str]]:
    categories: List[str] = []
    seen = set()
    for raw in _as_list(value):
        text = _canonical_category_text(raw)
        if _is_missing_value_label(text):
            continue
        key = _category_equivalence_key(text)
        if not text or key in seen:
            continue
        seen.add(key)
        categories.append(text)
    return categories or None


def _canonical_value_aliases(
    value_aliases: Any,
    categories: Sequence[str],
) -> Optional[Dict[str, List[str]]]:
    if not isinstance(value_aliases, dict):
        return None
    by_key = {_category_equivalence_key(category): category for category in categories}
    result: Dict[str, List[str]] = {}
    for raw_key, raw_aliases in value_aliases.items():
        category = by_key.get(_category_equivalence_key(raw_key))
        if category is None:
            continue
        aliases = []
        for alias in _as_list(raw_aliases):
            text = _canonical_category_text(alias)
            if text and not _is_missing_value_label(text):
                aliases.append(text)
        if aliases:
            result[category] = list(dict.fromkeys(aliases))
    return result or None


def _canonical_category_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).translate(_DASH_TRANSLATION)
    text = text.strip().replace("\u2265", ">=").replace("\u2264", "<=")
    text = " ".join(text.split())
    return text


def _category_equivalence_key(value: Any) -> str:
    text = _canonical_category_text(value).lower()
    text = text.replace("_", " ").replace(" ", "")
    return text


def _is_missing_value_label(value: Any) -> bool:
    key = _canonical_category_text(value).lower().replace("-", "_")
    key = " ".join(key.split())
    return key in _MISSING_VALUE_LABELS or key.replace(" ", "_") in _MISSING_VALUE_LABELS


def _append_value_policy(
    description: str,
    *,
    categories: Optional[Sequence[str]],
    value_aliases: Any,
    continuous: bool,
) -> str:
    base = description.strip()
    policy_parts: List[str] = []
    if continuous:
        policy_parts.append(
            "Value policy: return a numeric value only; return null for unknown, "
            "not reported, not assessed, unavailable, or qualitative-only values "
            "such as high/low without a numeric value."
        )
    else:
        cats = ", ".join(str(cat) for cat in categories or [])
        policy_parts.append(
            f"Value policy: return exactly one of [{cats}]; return null for "
            "unknown, not reported, not assessed, not tested, or unavailable values."
        )
        alias_text = _format_value_aliases(value_aliases, categories or [])
        if alias_text:
            policy_parts.append(f"Map value aliases as follows: {alias_text}.")

    policy = " ".join(policy_parts)
    if not base:
        return policy
    if "Value policy:" in base:
        return base
    return f"{base} {policy}"


def _format_value_aliases(value_aliases: Any, categories: Sequence[str]) -> str:
    if not isinstance(value_aliases, dict):
        return ""
    allowed = set(categories)
    chunks: List[str] = []
    for category, raw_aliases in value_aliases.items():
        if category not in allowed:
            continue
        aliases = [
            _canonical_category_text(alias)
            for alias in _as_list(raw_aliases)
            if _canonical_category_text(alias) and not _is_missing_value_label(alias)
        ]
        if aliases:
            chunks.append(f"{category}: {', '.join(dict.fromkeys(aliases))}")
    return "; ".join(chunks)


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


def _dedupe_agentic_specs(
    specs: Sequence[ExplicitFeatureSpec],
) -> List[ExplicitFeatureSpec]:
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
    return f"{left_text} {right_text}"


def _screened_spec_to_proposal(
    spec: ExplicitFeatureSpec,
    original: AgenticFeatureProposal,
) -> AgenticFeatureProposal:
    return AgenticFeatureProposal(
        action="add",
        name=spec.name,
        type=spec.type,
        categories=spec.categories,
        description=spec.description,
        roles=spec.roles,
        rationale=original.rationale,
        expected_signal=original.expected_signal,
    )


def _canonicalize_broad_screen_prepared_folds(
    initial_specs: Sequence[ExplicitFeatureSpec],
    prepared_folds: Sequence[BroadScreenPreparedFold],
) -> List[ExplicitFeatureSpec]:
    """Canonicalize the broad extraction universe and rewrite fold proposals."""
    initial_by_name = {spec.name: spec for spec in initial_specs}
    canonical_by_name: Dict[str, ExplicitFeatureSpec] = dict(initial_by_name)
    broad_order: List[str] = []
    merged_roles: Dict[str, List[str]] = {
        spec.name: list(spec.roles or []) for spec in initial_specs
    }

    for prepared in prepared_folds:
        for proposal in prepared.proposals:
            name = proposal.name
            if name in initial_by_name:
                continue
            proposed_spec = _proposal_to_spec(proposal)
            canonical = canonical_by_name.get(name)
            if canonical is None:
                canonical_by_name[name] = proposed_spec
                broad_order.append(name)
                merged_roles[name] = list(proposed_spec.roles or [])
                continue

            merged_roles[name] = _merge_roles(
                merged_roles.get(name, []),
                proposed_spec.roles or [],
            )
            if _spec_extraction_contract_key(canonical) != _spec_extraction_contract_key(
                proposed_spec
            ):
                logger.warning(
                    "Broad-screen proposal name %s appeared with multiple extraction "
                    "contracts; the first contract will be reused for all folds",
                    name,
                )

    for name in broad_order:
        canonical = canonical_by_name[name]
        canonical_by_name[name] = ExplicitFeatureSpec(
            name=canonical.name,
            type=canonical.type,
            categories=canonical.categories,
            description=canonical.description,
            value_aliases=getattr(canonical, "value_aliases", None),
            roles=merged_roles.get(name, canonical.roles or []),
        )

    for prepared in prepared_folds:
        rewritten: List[AgenticFeatureProposal] = []
        seen_names = set()
        for proposal in prepared.proposals:
            name = proposal.name
            if name in initial_by_name or name in seen_names:
                continue
            canonical = canonical_by_name.get(name)
            if canonical is None:
                continue
            rewritten.append(_proposal_from_spec(canonical, proposal))
            seen_names.add(name)
        prepared.proposals = rewritten

    return _dedupe_feature_specs(
        [
            *initial_specs,
            *[canonical_by_name[name] for name in broad_order],
        ]
    )


def _merge_roles(left: Sequence[str], right: Sequence[str]) -> List[str]:
    merged: List[str] = []
    for role in [*left, *right]:
        if role not in VALID_ROLES or role in merged:
            continue
        merged.append(role)
    return merged


def _enrich_broad_selection_proposals(
    raw_proposals: Sequence[Dict[str, Any]],
    available_by_name: Dict[str, ExplicitFeatureSpec],
) -> List[Dict[str, Any]]:
    """Fill contracts for add-by-name selections from the extracted shortlist."""
    enriched: List[Dict[str, Any]] = []
    for raw in raw_proposals:
        if not isinstance(raw, dict):
            enriched.append(raw)
            continue
        action = str(raw.get("action", "")).strip().lower()
        name = _normalize_feature_name(raw.get("name", ""))
        spec = available_by_name.get(name)
        if action != "add" or spec is None:
            enriched.append(dict(raw))
            continue
        filled = dict(raw)
        filled["action"] = "add"
        filled["name"] = spec.name
        filled["type"] = spec.type
        filled["categories"] = spec.categories
        filled["roles"] = list(spec.roles or [])
        filled["description"] = spec.description
        enriched.append(filled)
    return enriched


def _available_extracted_feature_payloads(
    train_df: pd.DataFrame,
    available_items: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    for item in available_items:
        spec = item.get("screened_spec")
        if not isinstance(spec, ExplicitFeatureSpec):
            continue
        extraction_summary = summarize_extractions(train_df, [spec])[0]
        payloads.append(
            {
                "rank": item.get("rank"),
                "name": spec.name,
                "type": spec.type,
                "categories": spec.categories,
                "roles": spec.roles,
                "description": spec.description,
                "coverage": extraction_summary.get("coverage"),
                "top_values": extraction_summary.get("top_values", {}),
                "screening_score": item.get("screening_score"),
                "confounder_score": item.get("confounder_score"),
                "modifier_score": item.get("modifier_score"),
                "role_diagnostics": _candidate_feedback_role_diagnostics(
                    {"role_diagnostics": item.get("role_diagnostics", [])}
                ),
            }
        )
    return payloads


def compare_candidate_to_baseline(
    baseline_rows: List[Dict[str, Any]],
    candidate_rows: List[Dict[str, Any]],
    search_config: AgenticFeatureSearchConfig,
) -> Dict[str, Any]:
    """Compare candidate inner-CV metrics to the current baseline feature set."""
    base = aggregate_metric_rows(baseline_rows)
    cand = aggregate_metric_rows(candidate_rows)
    base_r = base.get("r_loss_mean")
    cand_r = cand.get("r_loss_mean")
    if base_r is None or cand_r is None:
        r_loss_improvement = 0.0
    else:
        r_loss_improvement = (base_r - cand_r) / max(abs(base_r), 1e-8)

    outcome_delta = _metric_delta(cand, base, "outcome_auroc_mean")
    treatment_delta = _metric_delta(cand, base, "treatment_auroc_mean")
    improved_fold_fraction = _improved_fold_fraction(
        baseline_rows,
        candidate_rows,
        metric="r_loss",
        lower_is_better=True,
    )
    passes = (
        r_loss_improvement >= search_config.min_r_loss_improvement
        and outcome_delta >= -search_config.max_outcome_auroc_drop
        and treatment_delta >= -search_config.max_treatment_auroc_drop
        and improved_fold_fraction >= search_config.min_improvement_fold_fraction
    )
    return {
        "r_loss_improvement": float(r_loss_improvement),
        "outcome_auroc_delta": float(outcome_delta),
        "treatment_auroc_delta": float(treatment_delta),
        "improved_fold_fraction": float(improved_fold_fraction),
        "passes_acceptance": bool(passes),
        "baseline": _non_oracle_metrics(base),
        "candidate": _non_oracle_metrics(cand),
    }


def evaluate_candidate_role_diagnostics(
    dataset: pd.DataFrame,
    current_specs: List[ExplicitFeatureSpec],
    candidate_specs: List[ExplicitFeatureSpec],
    config: AppliedInferenceConfig,
    search_config: AgenticFeatureSearchConfig,
) -> List[Dict[str, Any]]:
    """Evaluate proposed feature roles using train-fold regressions.

    For each candidate feature, regress treatment and outcome on the current
    confounder-role features plus the candidate main effect. Then regress
    outcome on the same terms plus treatment-by-candidate interactions. These
    diagnostics are advisory and train-only; nested CV still decides acceptance.
    """
    if not getattr(search_config, "role_diagnostics_enabled", True):
        return []
    if not candidate_specs:
        return []

    diagnostics = []
    for candidate_spec in candidate_specs:
        diagnostics.append(
            _evaluate_single_candidate_role_diagnostic(
                dataset=dataset,
                current_specs=current_specs,
                candidate_spec=candidate_spec,
                config=config,
                search_config=search_config,
            )
        )
    return diagnostics


def screen_agentic_candidate_specs(
    dataset: pd.DataFrame,
    current_specs: List[ExplicitFeatureSpec],
    proposals: Sequence[AgenticFeatureProposal],
    config: AppliedInferenceConfig,
    search_config: AgenticFeatureSearchConfig,
) -> List[Dict[str, Any]]:
    """Screen broad candidate proposals with train-fold role diagnostics."""
    screened: List[Dict[str, Any]] = []
    for original_index, proposal in enumerate(proposals):
        proposed_spec = _proposal_to_spec(proposal)
        coverage_failures = _coverage_failures(
            dataset,
            [proposed_spec],
            search_config.min_feature_coverage,
        )
        diagnostics = evaluate_candidate_role_diagnostics(
            dataset=dataset,
            current_specs=current_specs,
            candidate_specs=[proposed_spec],
            config=config,
            search_config=search_config,
        )
        diagnostic = diagnostics[0] if diagnostics else {}
        recommended_roles = [
            role for role in diagnostic.get("recommended_roles", []) if role in VALID_ROLES
        ]
        confounder_score, modifier_score, screening_score = _role_screen_scores(diagnostic)
        rejection_reason = None
        if coverage_failures:
            rejection_reason = "low_feature_coverage"
        elif not diagnostic:
            rejection_reason = "missing_role_diagnostic"
        elif diagnostic.get("status") != "ok":
            rejection_reason = str(diagnostic.get("status", "role_diagnostic_failed"))
        elif not recommended_roles:
            rejection_reason = "no_role_signal"

        screened_spec = None
        if rejection_reason is None:
            screened_spec = ExplicitFeatureSpec(
                name=proposed_spec.name,
                type=proposed_spec.type,
                categories=proposed_spec.categories,
                description=proposed_spec.description,
                value_aliases=getattr(proposed_spec, "value_aliases", None),
                roles=recommended_roles,
            )

        screened.append(
            {
                "candidate_id": proposed_spec.name,
                "proposal": proposal,
                "proposed_spec": proposed_spec,
                "screened_spec": screened_spec,
                "role_diagnostics": diagnostics,
                "coverage_failures": coverage_failures,
                "screening_rejection_reason": rejection_reason,
                "confounder_score": confounder_score,
                "modifier_score": modifier_score,
                "screening_score": screening_score,
                "original_index": original_index,
                "kept_for_cv": False,
                "cv_accepted": False,
            }
        )

    ranked = sorted(
        screened,
        key=lambda item: (
            item["screening_rejection_reason"] is not None,
            -float(item["screening_score"]),
            int(item["original_index"]),
        ),
    )
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
    return ranked


def select_screened_candidates(
    screened: Sequence[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    """Select a balanced high-signal shortlist for inner-CV refinement."""
    top_k = max(1, int(top_k))
    eligible = [item for item in screened if item.get("screened_spec") is not None]
    eligible = sorted(
        eligible,
        key=lambda item: (
            -float(item.get("screening_score", 0.0)),
            int(item.get("rank", 0)),
        ),
    )
    selected: List[Dict[str, Any]] = []
    selected_ids = set()
    per_role_quota = min(10, max(1, top_k // 2))

    def add_for_role(role: str) -> None:
        for item in eligible:
            if len(selected) >= top_k:
                return
            spec = item.get("screened_spec")
            if spec is None or role not in spec.roles or item["candidate_id"] in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(item["candidate_id"])
            if (
                sum(1 for selected_item in selected if role in selected_item["screened_spec"].roles)
                >= per_role_quota
            ):
                return

    add_for_role("confounder")
    add_for_role("effect_modifier")
    for item in eligible:
        if len(selected) >= top_k:
            break
        if item["candidate_id"] in selected_ids:
            continue
        selected.append(item)
        selected_ids.add(item["candidate_id"])

    return sorted(selected, key=lambda item: int(item.get("rank", 0)))


def _role_screen_scores(diagnostic: Dict[str, Any]) -> Tuple[float, float, float]:
    treatment_delta = _diagnostic_score_delta(diagnostic, "treatment_association")
    outcome_delta = _diagnostic_score_delta(diagnostic, "outcome_association")
    interaction_delta = _diagnostic_score_delta(diagnostic, "treatment_interaction")
    confounder_score = (
        min(treatment_delta, outcome_delta)
        if treatment_delta is not None and outcome_delta is not None
        else 0.0
    )
    modifier_score = interaction_delta if interaction_delta is not None else 0.0
    return (
        float(max(confounder_score, 0.0)),
        float(max(modifier_score, 0.0)),
        float(max(confounder_score, modifier_score, 0.0)),
    )


def _diagnostic_score_delta(
    diagnostic: Dict[str, Any],
    key: str,
) -> Optional[float]:
    section = diagnostic.get(key)
    if not isinstance(section, dict):
        return None
    delta = section.get("score_delta")
    if not _is_number(delta):
        return None
    return float(delta)


def _evaluate_single_candidate_role_diagnostic(
    dataset: pd.DataFrame,
    current_specs: List[ExplicitFeatureSpec],
    candidate_spec: ExplicitFeatureSpec,
    config: AppliedInferenceConfig,
    search_config: AgenticFeatureSearchConfig,
) -> Dict[str, Any]:
    df = dataset.reset_index(drop=True)
    coverage = _feature_coverage(df, candidate_spec.name)
    non_missing_n = int(round(coverage * len(df)))
    base_payload = {
        "name": candidate_spec.name,
        "proposed_roles": list(candidate_spec.roles),
        "type": candidate_spec.type,
        "n": int(len(df)),
        "coverage": float(coverage),
        "non_missing_n": non_missing_n,
        "adjustment": "current_confounders",
    }

    min_n = int(getattr(search_config, "role_diagnostic_min_n", 20))
    min_non_missing = int(getattr(search_config, "role_diagnostic_min_non_missing", 10))
    if len(df) < min_n:
        return {
            **base_payload,
            "status": "insufficient_sample",
            "recommended_roles": [],
        }
    if non_missing_n < min_non_missing:
        return {
            **base_payload,
            "status": "insufficient_non_missing",
            "recommended_roles": [],
        }

    current_confounders = [
        spec
        for spec in current_specs
        if spec.name != candidate_spec.name and "confounder" in spec.roles
    ]
    _, w_matrix, _, w_names, _, _ = _build_features(df, current_confounders)
    z_spec = ExplicitFeatureSpec(
        name=candidate_spec.name,
        type=candidate_spec.type,
        categories=candidate_spec.categories,
        description=candidate_spec.description,
        value_aliases=getattr(candidate_spec, "value_aliases", None),
        roles=["confounder"],
    )
    _, z_matrix, _, z_names, _, _ = _build_features(df, [z_spec])

    n_rows = len(df)
    w_matrix = _feature_matrix_or_empty(w_matrix, n_rows)
    z_matrix = _feature_matrix_or_empty(z_matrix, n_rows)
    if z_matrix.shape[1] == 0 or not _has_any_variation(z_matrix):
        return {
            **base_payload,
            "status": "constant_candidate",
            "covariate_feature_names": w_names,
            "candidate_feature_names": z_names,
            "recommended_roles": [],
        }

    treatment = np.asarray(df[config.treatment_column].values).flatten()
    outcome = np.asarray(df[config.outcome_column].values).flatten()
    treatment_col = treatment.astype(float).reshape(-1, 1)

    treatment_diag = _nested_regression_diagnostic(
        base_x=w_matrix,
        full_x=np.hstack([w_matrix, z_matrix]),
        target=treatment,
        target_kind="binary",
        block_start=w_matrix.shape[1],
        block_width=z_matrix.shape[1],
    )

    outcome_base_x = np.hstack([w_matrix, treatment_col])
    outcome_main_x = np.hstack([outcome_base_x, z_matrix])
    outcome_diag = _nested_regression_diagnostic(
        base_x=outcome_base_x,
        full_x=outcome_main_x,
        target=outcome,
        target_kind=config.outcome_type,
        block_start=outcome_base_x.shape[1],
        block_width=z_matrix.shape[1],
    )

    interaction_matrix = z_matrix * treatment_col
    interaction_full_x = np.hstack([outcome_main_x, interaction_matrix])
    interaction_diag = _nested_regression_diagnostic(
        base_x=outcome_main_x,
        full_x=interaction_full_x,
        target=outcome,
        target_kind=config.outcome_type,
        block_start=outcome_main_x.shape[1],
        block_width=interaction_matrix.shape[1],
    )

    threshold = float(getattr(search_config, "role_diagnostic_score_delta_threshold", 0.001))
    treatment_signal = _score_delta_at_least(treatment_diag, threshold)
    outcome_signal = _score_delta_at_least(outcome_diag, threshold)
    interaction_signal = _score_delta_at_least(interaction_diag, threshold)
    recommended_roles = []
    if treatment_signal and outcome_signal:
        recommended_roles.append("confounder")
    if interaction_signal:
        recommended_roles.append("effect_modifier")

    return {
        **base_payload,
        "status": "ok",
        "covariate_feature_names": w_names,
        "candidate_feature_names": z_names,
        "treatment_association": treatment_diag,
        "outcome_association": outcome_diag,
        "treatment_interaction": interaction_diag,
        "confounder_signal": bool(treatment_signal and outcome_signal),
        "effect_modifier_signal": bool(interaction_signal),
        "recommended_roles": recommended_roles,
    }


def _nested_regression_diagnostic(
    base_x: np.ndarray,
    full_x: np.ndarray,
    target: np.ndarray,
    target_kind: str,
    block_start: int,
    block_width: int,
) -> Dict[str, Any]:
    target_kind = "continuous" if target_kind == "continuous" else "binary"
    y = np.asarray(target).flatten()
    base_x = np.asarray(base_x, dtype=np.float64)
    full_x = np.asarray(full_x, dtype=np.float64)
    finite = np.isfinite(y.astype(float, copy=False))
    finite &= np.all(np.isfinite(base_x), axis=1)
    finite &= np.all(np.isfinite(full_x), axis=1)
    if finite.sum() < 2:
        return {"status": "insufficient_finite_target", "target_kind": target_kind}

    base_x = base_x[finite]
    full_x = full_x[finite]
    y = y[finite]

    if target_kind == "binary":
        unique = np.unique(y)
        if len(unique) < 2:
            return {"status": "constant_target", "target_kind": target_kind}
        y_model = (y == unique[-1]).astype(int)
        base_pred, base_score, base_auroc, _ = _fit_binary_regression(base_x, y_model)
        full_pred, full_score, full_auroc, full_coef = _fit_binary_regression(full_x, y_model)
        del base_pred, full_pred
        result = {
            "status": "ok",
            "target_kind": target_kind,
            "score_metric": "neg_log_loss",
            "base_score": base_score,
            "full_score": full_score,
            "score_delta": _safe_subtract(full_score, base_score),
            "base_auroc": base_auroc,
            "full_auroc": full_auroc,
            "auroc_delta": _safe_subtract(full_auroc, base_auroc),
        }
    else:
        y_model = y.astype(float)
        if np.std(y_model) == 0:
            return {"status": "constant_target", "target_kind": target_kind}
        base_pred, base_score, _ = _fit_continuous_regression(base_x, y_model)
        full_pred, full_score, full_coef = _fit_continuous_regression(full_x, y_model)
        del base_pred, full_pred
        result = {
            "status": "ok",
            "target_kind": target_kind,
            "score_metric": "r2",
            "base_score": base_score,
            "full_score": full_score,
            "score_delta": _safe_subtract(full_score, base_score),
        }

    block_coef = np.asarray(full_coef[block_start : block_start + block_width], dtype=float)
    result["coefficient_l2_norm"] = float(np.linalg.norm(block_coef))
    result["n_model_rows"] = int(len(y))
    return result


def _fit_binary_regression(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, float, Optional[float], np.ndarray]:
    x = _ensure_regression_columns(x)
    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    model.fit(x, y)
    pred = model.predict_proba(x)[:, 1]
    clipped = np.clip(pred, 1e-6, 1.0 - 1e-6)
    score = -float(log_loss(y, clipped, labels=[0, 1]))
    return pred, score, _safe_roc_auc(y, pred), model.coef_.reshape(-1)


def _fit_continuous_regression(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, float, np.ndarray]:
    x = _ensure_regression_columns(x)
    model = Ridge(alpha=1.0)
    model.fit(x, y)
    pred = model.predict(x)
    return pred, float(r2_score(y, pred)), np.asarray(model.coef_).reshape(-1)


def _ensure_regression_columns(x: np.ndarray) -> np.ndarray:
    if x.shape[1] == 0:
        return np.zeros((x.shape[0], 1), dtype=np.float64)
    return x


def _feature_matrix_or_empty(
    matrix: Optional[np.ndarray],
    n_rows: int,
) -> np.ndarray:
    if matrix is None:
        return np.zeros((n_rows, 0), dtype=np.float32)
    return np.asarray(matrix, dtype=np.float32)


def _has_any_variation(matrix: np.ndarray) -> bool:
    if matrix.shape[1] == 0:
        return False
    return bool(np.any(np.nanstd(matrix, axis=0) > 1e-12))


def _score_delta_at_least(diagnostic: Dict[str, Any], threshold: float) -> bool:
    delta = diagnostic.get("score_delta")
    return _is_number(delta) and float(delta) >= threshold


def _safe_subtract(left: Any, right: Any) -> Optional[float]:
    if left is None or right is None:
        return None
    if not (_is_number(left) and _is_number(right)):
        return None
    return float(left) - float(right)


def _feature_coverage(dataset: pd.DataFrame, name: str) -> float:
    value_col = f"explicit_feat_{name}"
    missing_col = f"{value_col}_missing"
    if value_col not in dataset.columns:
        return 0.0
    missing = (
        dataset[missing_col].astype(bool)
        if missing_col in dataset.columns
        else dataset[value_col].isna()
    )
    return float(1.0 - missing.mean())


def build_iteration_feedback(
    recent_decisions: List[Dict[str, Any]],
    search_config: AgenticFeatureSearchConfig,
) -> List[Dict[str, Any]]:
    """Distill prior decisions into compact feedback for the next agent prompt."""
    feedback: List[Dict[str, Any]] = []
    for event in recent_decisions:
        event_name = event.get("event")
        payload = event.get("payload")
        if event_name == "agent_proposals" and isinstance(payload, dict):
            for rejected in payload.get("rejected", []):
                if not isinstance(rejected, dict):
                    continue
                raw_proposal = rejected.get("proposal", {})
                feedback.append(
                    {
                        "iteration": event.get("iteration"),
                        "candidate_id": _proposal_feedback_id(raw_proposal),
                        "status": "validation_rejected",
                        "failed_checks": [str(rejected.get("reason", "validation_failed"))],
                        "proposals": [_proposal_feedback_summary(raw_proposal)],
                        "instruction": (
                            "Do not repeat this proposal unchanged; fix the validation "
                            "failure or propose a different variable."
                        ),
                    }
                )
        elif event_name == "candidate_evaluations" and isinstance(payload, list):
            for item in payload:
                if not isinstance(item, dict):
                    continue
                comparison = item.get("comparison", {})
                accepted = bool(item.get("accepted", False))
                passed = bool(comparison.get("passes_acceptance", False))
                if accepted:
                    status = "accepted"
                elif passed:
                    status = "not_selected"
                else:
                    status = "rejected"
                entry = {
                    "iteration": event.get("iteration"),
                    "candidate_id": item.get("candidate_id"),
                    "status": status,
                    "proposals": [
                        _proposal_feedback_summary(proposal)
                        for proposal in item.get("proposals", [])
                    ],
                    "metrics": _candidate_feedback_metrics(comparison),
                }
                role_diagnostics = _candidate_feedback_role_diagnostics(item.get("summary"))
                if role_diagnostics:
                    entry["role_diagnostics"] = role_diagnostics
                if accepted:
                    entry["instruction"] = (
                        "This candidate became the current baseline; build on it "
                        "unless later feedback indicates a problem."
                    )
                elif passed:
                    entry["failed_checks"] = ["passed_thresholds_but_lower_ranked"]
                    entry["instruction"] = (
                        "This candidate passed acceptance thresholds but was not "
                        "selected because another candidate had stronger R-loss "
                        "improvement."
                    )
                else:
                    entry["failed_checks"] = _candidate_failed_checks(
                        comparison,
                        search_config,
                    )
                    entry["instruction"] = (
                        "Do not repeat this candidate unchanged; propose a different "
                        "baseline variable, extraction target, or role that addresses "
                        "the failed_checks."
                    )
                feedback.append(entry)

    return feedback[-20:]


def _proposal_feedback_id(proposal: Any) -> str:
    if isinstance(proposal, dict):
        name = proposal.get("name")
        if name:
            return _normalize_feature_name(name)
        action = proposal.get("action")
        if action:
            return str(action)
    return str(proposal)


def _proposal_feedback_summary(proposal: Any) -> Dict[str, Any]:
    if not isinstance(proposal, dict):
        return {"raw": str(proposal)}
    return {
        key: proposal.get(key)
        for key in ["action", "name", "type", "roles", "description"]
        if proposal.get(key) is not None
    }


def _candidate_feedback_metrics(comparison: Any) -> Dict[str, Any]:
    if not isinstance(comparison, dict):
        return {}
    keys = [
        "r_loss_improvement",
        "outcome_auroc_delta",
        "treatment_auroc_delta",
        "improved_fold_fraction",
        "passes_acceptance",
        "rejection_reason",
        "coverage_failures",
    ]
    return {key: comparison[key] for key in keys if key in comparison}


def _candidate_feedback_role_diagnostics(summary: Any) -> List[Dict[str, Any]]:
    if not isinstance(summary, dict):
        return []
    diagnostics = summary.get("role_diagnostics")
    if not isinstance(diagnostics, list):
        return []
    compact = []
    for diagnostic in diagnostics:
        if not isinstance(diagnostic, dict):
            continue
        entry = {
            key: diagnostic.get(key)
            for key in [
                "name",
                "status",
                "proposed_roles",
                "recommended_roles",
                "confounder_signal",
                "effect_modifier_signal",
                "coverage",
                "non_missing_n",
            ]
            if diagnostic.get(key) is not None
        }
        treatment = diagnostic.get("treatment_association", {})
        outcome = diagnostic.get("outcome_association", {})
        interaction = diagnostic.get("treatment_interaction", {})
        if isinstance(treatment, dict):
            entry["treatment_score_delta"] = treatment.get("score_delta")
            entry["treatment_auroc_delta"] = treatment.get("auroc_delta")
        if isinstance(outcome, dict):
            entry["outcome_score_delta"] = outcome.get("score_delta")
            entry["outcome_auroc_delta"] = outcome.get("auroc_delta")
        if isinstance(interaction, dict):
            entry["interaction_score_delta"] = interaction.get("score_delta")
            entry["interaction_auroc_delta"] = interaction.get("auroc_delta")
        compact.append({key: value for key, value in entry.items() if value is not None})
    return compact


def _candidate_failed_checks(
    comparison: Any,
    search_config: AgenticFeatureSearchConfig,
) -> List[str]:
    if not isinstance(comparison, dict):
        return ["candidate_evaluation_missing"]

    failed = []
    rejection_reason = comparison.get("rejection_reason")
    if rejection_reason:
        failed.append(f"rejection_reason: {rejection_reason}")

    for item in comparison.get("coverage_failures", []) or []:
        if not isinstance(item, dict):
            continue
        coverage = item.get("coverage")
        name = item.get("name", "feature")
        if _is_number(coverage):
            failed.append(
                f"coverage {name} {float(coverage):.4g} "
                f"< required {search_config.min_feature_coverage:.4g}"
            )

    r_loss_improvement = comparison.get("r_loss_improvement")
    if (
        _is_number(r_loss_improvement)
        and float(r_loss_improvement) < search_config.min_r_loss_improvement
    ):
        failed.append(
            f"r_loss_improvement {float(r_loss_improvement):.4g} "
            f"< required {search_config.min_r_loss_improvement:.4g}"
        )

    outcome_delta = comparison.get("outcome_auroc_delta")
    outcome_floor = -search_config.max_outcome_auroc_drop
    if _is_number(outcome_delta) and float(outcome_delta) < outcome_floor:
        failed.append(
            f"outcome_auroc_delta {float(outcome_delta):.4g} " f"< allowed {outcome_floor:.4g}"
        )

    treatment_delta = comparison.get("treatment_auroc_delta")
    treatment_floor = -search_config.max_treatment_auroc_drop
    if _is_number(treatment_delta) and float(treatment_delta) < treatment_floor:
        failed.append(
            f"treatment_auroc_delta {float(treatment_delta):.4g} "
            f"< allowed {treatment_floor:.4g}"
        )

    improved_fold_fraction = comparison.get("improved_fold_fraction")
    if (
        _is_number(improved_fold_fraction)
        and float(improved_fold_fraction) < search_config.min_improvement_fold_fraction
    ):
        failed.append(
            f"improved_fold_fraction {float(improved_fold_fraction):.4g} "
            f"< required {search_config.min_improvement_fold_fraction:.4g}"
        )

    if not failed:
        failed.append("did_not_pass_acceptance_thresholds")
    return failed


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value)


def aggregate_metric_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate numeric split metrics as mean/std, ignoring oracle-only keys."""
    if not rows:
        return {}
    df = pd.DataFrame([_non_oracle_metrics(row) for row in rows])
    result: Dict[str, Any] = {}
    for col in df.columns:
        if col in {"outer_fold", "iteration", "inner_fold", "fold"}:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        values = values[np.isfinite(values)]
        if len(values) == 0:
            continue
        result[f"{col}_mean"] = float(values.mean())
        result[f"{col}_std"] = float(values.std(ddof=0))
    return result


def summarize_extractions(
    dataset: pd.DataFrame,
    specs: List[ExplicitFeatureSpec],
) -> List[Dict[str, Any]]:
    """Summarize extraction coverage and observed values for the current features."""
    summaries = []
    for spec in specs:
        value_col = f"explicit_feat_{spec.name}"
        missing_col = f"{value_col}_missing"
        if value_col not in dataset.columns:
            summaries.append({"name": spec.name, "coverage": 0.0, "top_values": {}})
            continue
        missing = (
            dataset[missing_col].astype(bool)
            if missing_col in dataset.columns
            else dataset[value_col].isna()
        )
        observed = dataset.loc[~missing, value_col]
        summaries.append(
            {
                "name": spec.name,
                "coverage": float(1.0 - missing.mean()),
                "top_values": observed.astype(str).value_counts().head(8).to_dict(),
            }
        )
    return summaries


def _clinical_text_examples(
    dataset: pd.DataFrame,
    text_column: str,
    *,
    n_examples: int | None,
    max_chars: int | None,
) -> List[str]:
    """Return complete sampled notes for agent context.

    ``max_chars`` remains in the signature so older configuration files still
    load, but it is intentionally nonbinding.  Prompt construction must not
    truncate or reject a selected clinical note based on its character count.
    """
    del max_chars
    if text_column not in dataset.columns or len(dataset) == 0:
        return []
    nonempty = [
        str(text) for text in dataset[text_column].fillna("").tolist() if str(text).strip()
    ]
    if n_examples == 0:
        return []
    if n_examples is not None:
        sample = dataset.loc[
            dataset[text_column].fillna("").astype(str).str.strip().ne("")
        ].sample(
            n=min(int(n_examples), len(nonempty)),
            random_state=17,
        )
        nonempty = [str(text) for text in sample[text_column].fillna("").tolist()]
    return nonempty


def _coverage_failures(
    dataset: pd.DataFrame,
    specs: List[ExplicitFeatureSpec],
    min_coverage: float,
) -> List[Dict[str, Any]]:
    failures = []
    for item in summarize_extractions(dataset, specs):
        if item["coverage"] < min_coverage:
            failures.append({"name": item["name"], "coverage": item["coverage"]})
    return failures


def _coerce_proposal(raw: Dict[str, Any]) -> AgenticFeatureProposal:
    action = str(raw.get("action", "")).strip().lower()
    name = _normalize_feature_name(raw.get("name", ""))
    roles = raw.get("roles") or []
    if isinstance(roles, str):
        roles = [roles]
    categories = raw.get("categories")
    if categories is not None:
        categories = [str(cat) for cat in categories]
    return AgenticFeatureProposal(
        action=action,
        name=name,
        type=raw.get("type"),
        categories=categories,
        description=raw.get("description"),
        roles=[str(role).strip() for role in roles],
        rationale=raw.get("rationale"),
        expected_signal=raw.get("expected_signal"),
    )


def _proposal_rejection_reason(
    proposal: AgenticFeatureProposal,
    current_names: set,
    allow_removals: bool,
) -> Optional[str]:
    if proposal.action not in VALID_ACTIONS:
        return "invalid_action"
    if proposal.action == "none":
        return None
    if not proposal.name or not re.match(r"^[a-z][a-z0-9_]*$", proposal.name):
        return "invalid_name"
    if proposal.action == "add":
        if proposal.name in current_names:
            return "duplicate_feature"
        if proposal.type not in VALID_TYPES:
            return "invalid_type"
        if not proposal.roles or set(proposal.roles) - VALID_ROLES:
            return "invalid_roles"
        if proposal.type == "categorical" and not proposal.categories:
            return "missing_categories"
        if proposal.type == "categorical" and len(proposal.categories or []) > 8:
            return "too_many_categories"
        if not proposal.description:
            return "missing_description"
    elif proposal.action in {"remove", "update_role"}:
        if not allow_removals:
            return "removal_or_role_update_not_allowed_yet"
        if proposal.name not in current_names:
            return "unknown_existing_feature"
        if proposal.action == "update_role" and (
            not proposal.roles or set(proposal.roles) - VALID_ROLES
        ):
            return "invalid_roles"
    return None


def _candidate_groups(
    proposals: List[AgenticFeatureProposal],
) -> List[Tuple[str, List[AgenticFeatureProposal]]]:
    groups = [(proposal.name, [proposal]) for proposal in proposals]
    if len(proposals) > 1:
        bundled = []
        seen_names = set()
        for proposal in proposals:
            if proposal.name in seen_names:
                continue
            bundled.append(proposal)
            seen_names.add(proposal.name)
        if len(bundled) > 1:
            groups.append(("bundle", bundled))
    return groups


def _candidate_proposal_specs(
    current_specs: List[ExplicitFeatureSpec],
    candidate_specs: List[ExplicitFeatureSpec],
    proposal_group: Sequence[AgenticFeatureProposal],
) -> List[ExplicitFeatureSpec]:
    """Return add/update specs touched by this proposal group."""
    current_by_name = {spec.name: spec for spec in current_specs}
    candidate_by_name = {spec.name: spec for spec in candidate_specs}
    proposal_specs = []
    seen = set()
    for proposal in proposal_group:
        if proposal.action not in {"add", "update_role"}:
            continue
        if proposal.name in seen:
            continue
        spec = candidate_by_name.get(proposal.name) or current_by_name.get(proposal.name)
        if spec is not None:
            proposal_specs.append(spec)
            seen.add(proposal.name)
    return proposal_specs


def _candidate_role_diagnostic_specs(
    current_specs: List[ExplicitFeatureSpec],
    candidate_specs: List[ExplicitFeatureSpec],
    proposal_group: Sequence[AgenticFeatureProposal],
) -> List[ExplicitFeatureSpec]:
    return _candidate_proposal_specs(current_specs, candidate_specs, proposal_group)


def _choose_accepted_candidate(candidate_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    passing = [item for item in candidate_results if item["comparison"].get("passes_acceptance")]
    if not passing:
        return None
    return max(
        passing,
        key=lambda item: item["comparison"].get("r_loss_improvement", 0.0),
    )


def _dedupe_feature_specs(specs: Sequence[ExplicitFeatureSpec]) -> List[ExplicitFeatureSpec]:
    deduped = []
    seen = set()
    for spec in specs:
        if spec.name in seen:
            continue
        seen.add(spec.name)
        deduped.append(spec)
    return deduped


def _spec_extraction_contract_dict(spec: ExplicitFeatureSpec) -> Dict[str, Any]:
    """Return only fields that affect the extraction prompt and parser."""
    return {
        "name": spec.name,
        "type": spec.type,
        "categories": spec.categories,
        "description": spec.description,
        "value_aliases": getattr(spec, "value_aliases", None),
        # Roles are omitted from the prompt; keep a stable placeholder because
        # ExtractionCache's generic hash currently includes this field.
        "roles": [],
    }


def _spec_extraction_contract_key(spec: ExplicitFeatureSpec) -> str:
    return json.dumps(
        _spec_extraction_contract_dict(spec),
        sort_keys=True,
        default=_json_default,
    )


def _screening_decision_payload(item: Dict[str, Any]) -> Dict[str, Any]:
    spec = item.get("screened_spec") or item.get("proposed_spec")
    proposal = item.get("proposal")
    variables = []
    if isinstance(spec, ExplicitFeatureSpec):
        source = proposal if isinstance(proposal, AgenticFeatureProposal) else None
        fallback = AgenticFeatureProposal(
            action="add",
            name=spec.name,
            type=spec.type,
            categories=spec.categories,
            description=spec.description,
            roles=spec.roles,
        )
        variables.append(asdict(_screened_spec_to_proposal(spec, source or fallback)))
    comparison = item.get("cv_comparison", {})
    return {
        "candidate_id": item.get("candidate_id"),
        "rank": item.get("rank"),
        "variables": variables,
        "screening_score": item.get("screening_score"),
        "confounder_score": item.get("confounder_score"),
        "modifier_score": item.get("modifier_score"),
        "kept_for_cv": bool(item.get("kept_for_cv", False)),
        "cv_accepted": bool(item.get("cv_accepted", False)),
        "passes_acceptance": bool(
            isinstance(comparison, dict) and comparison.get("passes_acceptance", False)
        ),
        "screening_rejection_reason": item.get("screening_rejection_reason"),
        "coverage_failures": item.get("coverage_failures", []),
        "role_diagnostics": item.get("role_diagnostics", []),
    }


def _screening_metric_row(
    outer_fold: int,
    iteration: int,
    item: Dict[str, Any],
) -> Dict[str, Any]:
    diagnostic = item.get("role_diagnostics", [{}])[0] if item.get("role_diagnostics") else {}
    comparison = item.get("cv_comparison", {})
    screened_spec = item.get("screened_spec")
    proposed_spec = item.get("proposed_spec")
    spec = screened_spec or proposed_spec
    return {
        "outer_fold": outer_fold,
        "iteration": iteration,
        "candidate_id": item.get("candidate_id"),
        "rank": item.get("rank"),
        "kept_for_cv": bool(item.get("kept_for_cv", False)),
        "cv_accepted": bool(item.get("cv_accepted", False)),
        "screening_rejection_reason": item.get("screening_rejection_reason"),
        "coverage": diagnostic.get("coverage"),
        "diagnostic_status": diagnostic.get("status"),
        "proposed_roles": ",".join(getattr(spec, "roles", []) or []),
        "recommended_roles": ",".join(
            diagnostic.get("recommended_roles", []) if isinstance(diagnostic, dict) else []
        ),
        "screening_score": item.get("screening_score"),
        "confounder_score": item.get("confounder_score"),
        "modifier_score": item.get("modifier_score"),
        "treatment_score_delta": _diagnostic_score_delta(
            diagnostic,
            "treatment_association",
        ),
        "outcome_score_delta": _diagnostic_score_delta(
            diagnostic,
            "outcome_association",
        ),
        "interaction_score_delta": _diagnostic_score_delta(
            diagnostic,
            "treatment_interaction",
        ),
        "r_loss_improvement": (
            comparison.get("r_loss_improvement") if isinstance(comparison, dict) else None
        ),
        "passes_acceptance": (
            comparison.get("passes_acceptance") if isinstance(comparison, dict) else None
        ),
    }


def _make_splits(
    df: pd.DataFrame,
    config: AppliedInferenceConfig,
    n_splits: int,
    random_state: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if n_splits > len(df):
        raise ValueError(f"n_splits={n_splits} exceeds n={len(df)}")
    y = df[config.treatment_column].astype(str) + "_" + df[config.outcome_column].astype(str)
    counts = y.value_counts()
    if len(counts) >= 2 and counts.min() >= n_splits:
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        return list(splitter.split(df, y))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(splitter.split(df))


def _fit_predict_propensity(
    train_x: np.ndarray,
    train_t: np.ndarray,
    test_x: np.ndarray,
    cf_config: ExplicitFeatureForestConfig,
    random_state: int,
) -> np.ndarray:
    if len(np.unique(train_t)) < 2:
        return np.full(len(test_x), float(train_t[0]), dtype=np.float32)
    model = RandomForestClassifier(
        n_estimators=max(50, cf_config.n_estimators // 2),
        max_depth=cf_config.max_depth,
        min_samples_leaf=cf_config.min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(train_x, train_t)
    return model.predict_proba(test_x)[:, 1]


def _fit_predict_outcome(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    outcome_type: str,
    cf_config: ExplicitFeatureForestConfig,
    random_state: int,
) -> np.ndarray:
    if outcome_type == "continuous":
        model = RandomForestRegressor(
            n_estimators=max(50, cf_config.n_estimators // 2),
            max_depth=cf_config.max_depth,
            min_samples_leaf=cf_config.min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(train_x, train_y)
        return model.predict(test_x)
    if len(np.unique(train_y)) < 2:
        return np.full(len(test_x), float(train_y[0]), dtype=np.float32)
    model = RandomForestClassifier(
        n_estimators=max(50, cf_config.n_estimators // 2),
        max_depth=cf_config.max_depth,
        min_samples_leaf=cf_config.min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(train_x, train_y)
    return model.predict_proba(test_x)[:, 1]


def _r_loss(
    y: np.ndarray,
    t: np.ndarray,
    outcome_pred: np.ndarray,
    propensity: np.ndarray,
    tau: np.ndarray,
) -> float:
    residual_y = np.asarray(y) - np.asarray(outcome_pred)
    residual_t = np.asarray(t) - np.asarray(propensity)
    return float(np.mean((residual_y - np.asarray(tau) * residual_t) ** 2))


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return None


def _safe_corr(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _metric_delta(candidate: Dict[str, Any], baseline: Dict[str, Any], key: str) -> float:
    cand = candidate.get(key)
    base = baseline.get(key)
    if cand is None or base is None:
        return 0.0
    return float(cand - base)


def _improved_fold_fraction(
    baseline_rows: List[Dict[str, Any]],
    candidate_rows: List[Dict[str, Any]],
    metric: str,
    lower_is_better: bool,
) -> float:
    baseline_by_fold = {row.get("inner_fold", row.get("fold")): row for row in baseline_rows}
    candidate_by_fold = {row.get("inner_fold", row.get("fold")): row for row in candidate_rows}
    common = sorted(set(baseline_by_fold) & set(candidate_by_fold))
    if not common:
        return 0.0
    improved = 0
    for fold in common:
        base = baseline_by_fold[fold].get(metric)
        cand = candidate_by_fold[fold].get(metric)
        if base is None or cand is None:
            continue
        improved += cand < base if lower_is_better else cand > base
    return improved / len(common)


def _without_list_values(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value for key, value in metrics.items() if not isinstance(value, (list, dict, tuple))
    }


def _non_oracle_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if not str(key).startswith("oracle_") and not str(key).startswith("true_")
    }


def _clinical_question_text(config: AppliedInferenceConfig) -> str:
    configured = str(getattr(config, "clinical_question", "") or "").strip()
    if configured:
        return configured
    return "What is the causal effect of " f"{config.treatment_column} on {config.outcome_column}?"


def _scrub_decision_payload(payload: Any, save_agent_context: bool) -> Any:
    """Remove raw clinical text examples from persisted decision artifacts."""
    if save_agent_context:
        return payload
    if isinstance(payload, dict):
        scrubbed = {}
        for key, value in payload.items():
            if key == "clinical_text_examples":
                scrubbed[key] = []
            else:
                scrubbed[key] = _scrub_decision_payload(value, save_agent_context)
        return scrubbed
    if isinstance(payload, list):
        return [_scrub_decision_payload(item, save_agent_context) for item in payload]
    return payload


def _spec_to_dict(spec: ExplicitFeatureSpec) -> Dict[str, Any]:
    return {
        "name": spec.name,
        "type": spec.type,
        "categories": spec.categories,
        "description": spec.description,
        "value_aliases": getattr(spec, "value_aliases", None),
        "roles": spec.roles,
    }


def _spec_names(specs: List[ExplicitFeatureSpec]) -> List[str]:
    return [spec.name for spec in specs]


def _spec_signature(specs: Sequence[ExplicitFeatureSpec]) -> List[Tuple[Any, ...]]:
    return [
        (
            spec.name,
            spec.type,
            tuple(spec.categories or []),
            spec.description,
            json.dumps(getattr(spec, "value_aliases", None), sort_keys=True),
            tuple(spec.roles or []),
        )
        for spec in specs
    ]


def _normalize_feature_name(name: Any) -> str:
    name = str(name or "").strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, ExplicitFeatureSpec):
        return _spec_to_dict(value)
    return str(value)
