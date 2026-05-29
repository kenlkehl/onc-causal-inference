"""Agentic explicit-feature causal forest search.

This module runs an adaptive, LLM-guided variable search around the existing
explicit-feature causal forest. The reported performance comes from outer CV;
all feature-set decisions are made with inner CV on each outer-training split.
"""

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

from ..config import (
    AgenticFeatureSearchConfig,
    AppliedInferenceConfig,
    ExplicitFeatureForestConfig,
    ExplicitFeatureSpec,
)
from ..extraction import ExtractionCache, VLLMFeatureExtractor
from ..models.causal_forest_head import CausalForestHead
from .applied_explicit_feature_forest import _build_features, _hstack_present


logger = logging.getLogger(__name__)

AGENT_PROMPT_VERSION = "agentic_explicit_feature_search_v1"
EXTRACTION_PROMPT_VERSION = "explicit_features_v2"
VALID_ACTIONS = {"add", "remove", "update_role", "none"}
VALID_ROLES = {"confounder", "effect_modifier"}
VALID_TYPES = {"categorical", "continuous"}
POST_TREATMENT_LEAKAGE_TERMS = (
    "post-treatment",
    "post treatment",
    "after treatment",
    "after therapy",
    "after systemic therapy",
    "post therapy",
    "future imaging",
    "follow-up imaging",
)
OUTCOME_TARGET_TERMS = (
    "treatment response",
    "response to",
    "response category",
    "objective response",
    "radiographic response",
    "complete response",
    "partial response",
    "stable disease",
    "progressive disease",
    "survival",
    "progression-free",
    "progression free",
    "overall survival",
    "mortality",
    "death",
    "recurrence",
    "relapse",
    "toxicity",
    "adverse event",
)
RATIONALE_ONLY_LEAKAGE_TERMS = (
    "post-treatment",
    "post treatment",
    "after treatment",
    "after therapy",
    "future imaging",
    "survival after",
    "progression after",
    "toxicity after",
)
BASELINE_ALLOWED_TARGET_TERMS = (
    "age",
    "sex",
    "gender",
    "race",
    "ethnicity",
    "smoking",
    "pack year",
    "ecog",
    "performance status",
    "comorbidity",
    "histology",
    "stage",
    "tumor",
    "metastasis",
    "metastatic",
    "biomarker",
    "pdl1",
    "pd l1",
    "egfr",
    "alk",
    "kras",
    "ros1",
    "braf",
    "met",
    "ret",
    "ntrk",
    "lab",
    "laboratory",
    "neutrophil",
    "lymphocyte",
    "albumin",
    "hemoglobin",
    "creatinine",
    "ldh",
)
BASELINE_ANCHOR_TERMS = (
    "baseline",
    "pre-treatment",
    "pre treatment",
    "pretreatment",
    "before treatment",
    "prior to treatment",
    "at diagnosis",
    "at presentation",
    "treatment initiation",
)


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
    leakage_risk: str = "low"


@dataclass
class SplitEvaluation:
    """Predictions and metrics for one train/test split."""

    predictions: pd.DataFrame
    metrics: Dict[str, Any]


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
            logger.info(
                "Agentic explicit-feature search is starting from an empty feature set"
            )

        self.proposal_agent = proposal_agent or OpenAICompatibleFeatureSearchAgent(
            self.search_config
        )
        self.extraction_provider = extraction_provider or VLLMExplicitFeatureExtractionProvider(
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

    def run(self) -> None:
        """Execute outer CV, inner adaptive search, and final reporting."""
        logger.info("=" * 80)
        logger.info("AGENTIC EXPLICIT FEATURE CAUSAL FOREST")
        logger.info("=" * 80)

        # Ensure initial variables are available before the first inner search.
        self.dataset = self.extraction_provider.ensure_features(self.dataset, self.initial_specs)

        outer_splits = _make_splits(
            self.dataset,
            self.config,
            n_splits=self.search_config.outer_folds,
            random_state=self.search_config.random_state,
        )

        outer_predictions = []
        for outer_fold, (train_idx, test_idx) in enumerate(outer_splits, start=1):
            logger.info(
                "Outer fold %s/%s: train=%s test=%s",
                outer_fold,
                len(outer_splits),
                len(train_idx),
                len(test_idx),
            )
            selected_specs = self._search_outer_train(outer_fold, train_idx)
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
            raw_proposals = self.proposal_agent.propose(context)
            proposals, rejected = validate_agentic_proposals(
                raw_proposals,
                current_specs=current_specs,
                search_config=self.search_config,
                allow_removals=accepted_additions > 0,
            )
            self._record_decision(
                outer_fold,
                iteration,
                "agent_proposals",
                {
                    "raw_count": len(raw_proposals),
                    "valid_count": len(proposals),
                    "rejected": rejected,
                    "context": context,
                },
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
                coverage_failures = _coverage_failures(
                    self.dataset.iloc[outer_train_idx],
                    candidate_specs,
                    self.search_config.min_feature_coverage,
                )
                if coverage_failures:
                    candidate_results.append(
                        {
                            "candidate_id": candidate_id,
                            "proposal_group": proposal_group,
                            "specs": candidate_specs,
                            "rows": [],
                            "summary": {"coverage_failures": coverage_failures},
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
            "outcome_type": self.config.outcome_type,
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
        with open(self.artifact_dir / "feature_sets.json", "w") as f:
            json.dump(self.feature_set_rows, f, indent=2, default=_json_default)
        with open(self.artifact_dir / "agent_decisions.jsonl", "w") as f:
            for event in self.decision_events:
                f.write(json.dumps(event, default=_json_default) + "\n")
        logger.info("Agentic search artifacts saved to: %s", self.artifact_dir)


class OpenAICompatibleFeatureSearchAgent:
    """LLM proposal agent using an OpenAI-compatible chat completion endpoint."""

    def __init__(self, search_config: AgenticFeatureSearchConfig):
        self.search_config = search_config
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package is required for agentic feature proposals. "
                "Install the extraction extra or inject a custom proposal_agent."
            ) from exc
        self._client = OpenAI(
            base_url=self.search_config.agent_server_url,
            api_key=self.search_config.agent_api_key,
            max_retries=0,
        )

    def propose(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        self._ensure_client()
        prompt = build_agent_prompt(context, self.search_config)
        response = self._client.chat.completions.create(
            model=self.search_config.agent_model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.search_config.agent_temperature,
            max_tokens=self.search_config.agent_max_tokens,
        )
        content = response.choices[0].message.content or ""
        return parse_agent_response(content)


class VLLMExplicitFeatureExtractionProvider:
    """Ensure requested explicit feature columns exist, one variable at a time."""

    def __init__(self, config: AppliedInferenceConfig, output_dir: Path):
        self.config = config
        self.feature_config = config.explicit_features
        self.output_dir = Path(output_dir)
        self.cache = ExtractionCache(
            cache_dir=self.feature_config.cache_dir or str(self.output_dir)
        )

    def ensure_features(
        self,
        dataset: pd.DataFrame,
        specs: List[ExplicitFeatureSpec],
    ) -> pd.DataFrame:
        dataset = dataset.copy()
        for spec in specs:
            value_col = f"explicit_feat_{spec.name}"
            missing_col = f"{value_col}_missing"
            if value_col in dataset.columns and missing_col in dataset.columns:
                continue
            extracted_df = self._extract_one_spec(dataset, spec)
            for col in extracted_df.columns:
                dataset[col] = extracted_df[col].values
        return dataset

    def _extract_one_spec(self, dataset: pd.DataFrame, spec: ExplicitFeatureSpec) -> pd.DataFrame:
        cache_config = {
            "features": [spec],
            "prompt_template_version": EXTRACTION_PROMPT_VERSION,
            "vllm_model_name": self.feature_config.vllm_model_name,
            "vllm_max_model_len": self.feature_config.vllm_max_model_len,
            "extraction_temperature": self.feature_config.extraction_temperature,
            "extraction_max_tokens": self.feature_config.extraction_max_tokens,
            "extraction_max_text_length": self.feature_config.extraction_max_text_length,
        }
        cached = None
        if self.feature_config.cache_enabled:
            cached = self.cache.load_if_valid(
                self.config.dataset_path or "in_memory_dataset",
                cache_config,
                expected_rows=len(dataset),
            )
        if cached is not None:
            return cached

        logger.info("Extracting agentic feature with LLM: %s", spec.name)
        extractor = VLLMFeatureExtractor(
            specs=[spec],
            mode=self.feature_config.vllm_mode,
            server_url=self.feature_config.vllm_server_url or "http://localhost:8000/v1",
            model_name=self.feature_config.vllm_model_name,
            tensor_parallel_size=self.feature_config.vllm_tensor_parallel_size,
            gpu_memory_utilization=self.feature_config.vllm_gpu_memory_utilization,
            download_dir=self.feature_config.vllm_download_dir,
            max_model_len=self.feature_config.vllm_max_model_len,
            max_retries=self.feature_config.extraction_max_retries,
            temperature=self.feature_config.extraction_temperature,
            max_tokens=self.feature_config.extraction_max_tokens,
            max_text_length=self.feature_config.extraction_max_text_length,
        )
        try:
            extracted_df = extractor.extract_to_dataframe(
                dataset[self.config.text_column].tolist(),
                batch_size=self.feature_config.extraction_batch_size,
            )
        finally:
            extractor.cleanup()

        if self.feature_config.cache_enabled:
            self.cache.save(
                self.config.dataset_path or "in_memory_dataset",
                cache_config,
                extracted_df,
            )
        return extracted_df


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


def build_agent_prompt(
    context: Dict[str, Any],
    search_config: AgenticFeatureSearchConfig,
) -> str:
    """Construct the proposal prompt sent to the LLM agent."""
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
Propose only pre-treatment patient, tumor, disease, lab, biomarker, or baseline clinical variables that are plausibly extractable from the text and could improve confounding adjustment or CATE heterogeneity.

Do not propose post-treatment outcomes, treatment response, survival, toxicity after treatment, future imaging response, or variables that are descendants of treatment.
Baseline demographics and disease descriptors are allowed when measured at or before treatment, including age, sex, race/ethnicity, smoking history, ECOG/performance status, comorbidities, histology, stage, tumor burden/size, metastatic sites, baseline labs, molecular markers, and PD-L1 expression. Do not mark those baseline variables as leaky merely because they are broadly prognostic or may modify treatment response.
For age-like variables, define the target as age in years at baseline, diagnosis, presentation, or treatment initiation.

Return JSON only with this shape:
{{
  "proposals": [
    {{
      "action": "add|remove|update_role|none",
      "name": "snake_case_variable_name",
      "type": "categorical|continuous",
      "categories": ["category_a", "category_b"],
      "roles": ["confounder", "effect_modifier"],
      "description": "exact extraction target using pre-treatment information only",
      "rationale": "why this may help",
      "expected_signal": "treatment, outcome, or tau signal expected",
      "leakage_risk": "low|medium|high"
    }}
  ]
}}

Limits:
- At most {search_config.max_additions_per_iter} add proposals.
- At most {search_config.max_removals_per_iter} remove proposals.
- Use "none" if no defensible pre-treatment variable is available.
- For categorical variables, provide 2-8 mutually exclusive categories.
- Review iteration_feedback and recent_decisions before proposing. Do not repeat
  a rejected feature unchanged; if revisiting a rejected concept, change the
  extraction target, type/categories, or role to directly address failed_checks.

Current nested-CV context:
{context_json}
"""


def parse_agent_response(response: str) -> List[Dict[str, Any]]:
    """Parse JSON proposals from an LLM response."""
    response = response.strip()
    match = re.search(r"\{.*\}", response, re.DOTALL)
    json_str = match.group(0) if match else response
    parsed = json.loads(json_str)
    if isinstance(parsed, list):
        return parsed
    proposals = parsed.get("proposals", [])
    if not isinstance(proposals, list):
        raise ValueError("Agent response JSON must contain a proposals list")
    return proposals


def validate_agentic_proposals(
    raw_proposals: Sequence[Dict[str, Any]],
    current_specs: List[ExplicitFeatureSpec],
    search_config: AgenticFeatureSearchConfig,
    allow_removals: bool,
) -> Tuple[List[AgenticFeatureProposal], List[Dict[str, Any]]]:
    """Validate raw LLM proposals against schema, role, and leakage guards."""
    current_names = {spec.name for spec in current_specs}
    valid: List[AgenticFeatureProposal] = []
    rejected = []
    additions = 0
    removals = 0

    for raw in raw_proposals:
        try:
            proposal = _coerce_proposal(raw)
            reason = _proposal_rejection_reason(proposal, current_names, allow_removals)
            if reason is None and proposal.action == "add":
                additions += 1
                if additions > search_config.max_additions_per_iter:
                    reason = "too_many_additions"
            if reason is None and proposal.action == "remove":
                removals += 1
                if removals > search_config.max_removals_per_iter:
                    reason = "too_many_removals"
            if reason is None and proposal.action != "none":
                valid.append(proposal)
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
                            roles=proposal.roles,
                        )
                    )
                else:
                    updated.append(spec)
            specs = updated
    return specs


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
                            "failure or propose a different pre-treatment variable."
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
    return {
        key: comparison[key]
        for key in keys
        if key in comparison
    }


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
            f"outcome_auroc_delta {float(outcome_delta):.4g} "
            f"< allowed {outcome_floor:.4g}"
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
        missing = dataset[missing_col].astype(bool) if missing_col in dataset.columns else dataset[value_col].isna()
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
    n_examples: int = 3,
    max_chars: int = 1600,
) -> List[str]:
    if text_column not in dataset.columns or len(dataset) == 0:
        return []
    sample = dataset.sample(
        n=min(n_examples, len(dataset)),
        random_state=17,
    )
    return [
        str(text)[:max_chars]
        for text in sample[text_column].fillna("").tolist()
        if str(text).strip()
    ]


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
        leakage_risk=str(raw.get("leakage_risk", "low")).strip().lower(),
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
    leakage_reason = _proposal_leakage_reason(proposal)
    if leakage_reason is not None:
        return leakage_reason
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


def _proposal_leakage_reason(proposal: AgenticFeatureProposal) -> Optional[str]:
    """Return a leakage rejection reason, biased toward the extraction target.

    The LLM often explains baseline variables by saying they may influence
    "response to therapy." That rationale is not leakage by itself. The guard
    therefore focuses on the proposed variable name and extraction target, while
    still rejecting explicitly post-treatment rationale.
    """
    if proposal.leakage_risk == "high":
        return "high_leakage_risk"

    name_text = _normalize_leakage_text(proposal.name)
    description_text = _normalize_leakage_text(proposal.description)
    target_text = f"{name_text} {description_text}".strip()
    rationale_text = _normalize_leakage_text(
        " ".join(
            str(part or "")
            for part in [proposal.rationale, proposal.expected_signal]
        )
    )

    if _contains_any(target_text, POST_TREATMENT_LEAKAGE_TERMS):
        return "post_treatment_or_outcome_leakage"

    # If the variable name itself is an outcome/response concept, reject it
    # even if the LLM tries to prefix it with "baseline".
    if _contains_any(name_text, OUTCOME_TARGET_TERMS):
        return "post_treatment_or_outcome_leakage"

    if (
        _contains_any(target_text, OUTCOME_TARGET_TERMS)
        and not _is_allowed_baseline_target(name_text, target_text)
    ):
        return "post_treatment_or_outcome_leakage"

    if _contains_any(rationale_text, RATIONALE_ONLY_LEAKAGE_TERMS):
        return "post_treatment_or_outcome_leakage"

    return None


def _is_allowed_baseline_target(name_text: str, target_text: str) -> bool:
    has_baseline_anchor = _contains_any(target_text, BASELINE_ANCHOR_TERMS)
    has_safe_concept = _contains_any(name_text, BASELINE_ALLOWED_TARGET_TERMS)
    return has_baseline_anchor and has_safe_concept


def _contains_any(text: str, terms: Sequence[str]) -> bool:
    return any(_normalize_leakage_text(term) in text for term in terms)


def _normalize_leakage_text(text: Any) -> str:
    normalized = str(text or "").lower()
    normalized = re.sub(r"[_\-]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


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


def _choose_accepted_candidate(candidate_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    passing = [
        item for item in candidate_results if item["comparison"].get("passes_acceptance")
    ]
    if not passing:
        return None
    return max(
        passing,
        key=lambda item: item["comparison"].get("r_loss_improvement", 0.0),
    )


def _make_splits(
    df: pd.DataFrame,
    config: AppliedInferenceConfig,
    n_splits: int,
    random_state: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if n_splits > len(df):
        raise ValueError(f"n_splits={n_splits} exceeds n={len(df)}")
    y = (
        df[config.treatment_column].astype(str)
        + "_"
        + df[config.outcome_column].astype(str)
    )
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
        key: value
        for key, value in metrics.items()
        if not isinstance(value, (list, dict, tuple))
    }


def _non_oracle_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if not str(key).startswith("oracle_") and not str(key).startswith("true_")
    }


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
        return [
            _scrub_decision_payload(item, save_agent_context)
            for item in payload
        ]
    return payload


def _spec_to_dict(spec: ExplicitFeatureSpec) -> Dict[str, Any]:
    return {
        "name": spec.name,
        "type": spec.type,
        "categories": spec.categories,
        "description": spec.description,
        "roles": spec.roles,
    }


def _spec_names(specs: List[ExplicitFeatureSpec]) -> List[str]:
    return [spec.name for spec in specs]


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
