"""Agentic attention-evidence variable discovery plus explicit-feature forest."""

from __future__ import annotations

from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import logging
import os
import queue
import re
import threading
import multiprocessing as mp
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    log_loss,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold

from ..config import (
    HTR_NUMERIC_SUMMARY_KEYS,
    AppliedInferenceConfig,
    ExplicitFeatureForestConfig,
    ExplicitFeatureSpec,
    omit_inactive_htr_numeric_settings,
)
from .agentic_explicit_feature_forest import (
    AgenticFeatureProposal,
    OpenAICompatibleFeatureSearchAgent,
    VLLMExplicitFeatureExtractionProvider,
    _get_agent_response_trace,
    _proposal_agent_supports_value_harmonization,
    apply_agentic_value_harmonization,
    validate_agentic_proposals,
)
from .applied import drop_oracle_columns, standalone_outer_splits
from .applied_explicit_feature_forest import (
    _build_features,
    _hstack_present,
    add_forest_prediction_columns,
    fit_predict_explicit_forest,
)
from .htr_trainer import (
    EffectNet,
    HTRTrainer,
    NuisanceNet,
    cross_fitted_calibrators,
    effect_objective_name,
    r_pseudo_outcome as _r_pseudo_outcome,  # _crossfit_effect has a local of that name
)
from ..utils.calibration import (
    binary_calibration_metrics,
    clip_probability,
)

logger = logging.getLogger(__name__)

# Recorded in the nuisance checkpoint fingerprint so checkpoints written with
# the old in-sample calibration are never reused.
NUISANCE_CALIBRATION_SCOPE = "cross_fitted_out_of_fold_v1"

VALID_ROLES = {"confounder", "effect_modifier"}
VALID_TYPES = {"categorical", "continuous"}


def _running_inside_loky_worker() -> bool:
    try:
        current = mp.current_process()
        process_type = f"{type(current).__module__}.{type(current).__name__}"
        if "loky" in process_type.lower():
            return True
    except Exception:
        return False
    try:
        from joblib.externals.loky import process_executor

        return int(getattr(process_executor, "_CURRENT_DEPTH", 0) or 0) > 0
    except Exception:
        return False


def run_agentic_attention_variable_forest(
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    output_path: Path,
    device: Optional[torch.device] = None,
    num_workers: int = 1,
    gpu_ids: Optional[Sequence[int]] = None,
    devices: Optional[Sequence[torch.device | str]] = None,
    proposal_agent: Optional[Any] = None,
    extraction_provider: Optional[Any] = None,
) -> None:
    """Run the attention-evidence variable discovery forest pipeline."""
    runner = AgenticAttentionVariableForestRunner(
        dataset=dataset,
        config=config,
        output_path=output_path,
        device=device or torch.device("cpu"),
        num_workers=num_workers,
        gpu_ids=gpu_ids,
        devices=devices,
        proposal_agent=proposal_agent,
        extraction_provider=extraction_provider,
    )
    runner.run()


class AgenticAttentionVariableForestRunner(HTRTrainer):
    """End-to-end implementation of the proposed attention-variable strategy.

    The HTR networks, loaders, and training loops are inherited from
    :class:`HTRTrainer`; this runner adds the cross-fit orchestration, fold
    checkpoints, thread-local device pool, and DataLoader worker policy.
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        config: AppliedInferenceConfig,
        output_path: Path,
        device: torch.device,
        num_workers: int = 1,
        gpu_ids: Optional[Sequence[int]] = None,
        devices: Optional[Sequence[torch.device | str]] = None,
        proposal_agent: Optional[Any] = None,
        extraction_provider: Optional[Any] = None,
    ):
        self._thread_state = threading.local()
        # Sets ``config``, ``device`` (via the property setter) and ``avf_config``.
        super().__init__(config=config, device=device, num_workers=num_workers)
        self.dataset = drop_oracle_columns(dataset).reset_index(drop=True).copy()
        self.dataset["_oci_row_id"] = np.arange(len(self.dataset), dtype=int)
        self.output_path = Path(output_path)
        self.artifact_dir = self.output_path.parent / "agentic_attention_variable_forest"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.num_workers = 1 if num_workers is None else int(num_workers)
        self._has_custom_proposal_agent = proposal_agent is not None
        self._has_custom_extraction_provider = extraction_provider is not None
        self._has_external_components = (
            self._has_custom_proposal_agent or self._has_custom_extraction_provider
        )
        self.devices = self._resolve_device_pool(
            base_device=self._device,
            gpu_ids=gpu_ids,
            devices=devices,
        )
        self.agent_search_config = getattr(config.architecture, "agentic_feature_search")
        self.cf_config: ExplicitFeatureForestConfig = getattr(
            config.architecture,
            "explicit_feature_forest",
            ExplicitFeatureForestConfig(),
        )
        self.proposal_agent = proposal_agent or OpenAICompatibleFeatureSearchAgent(
            self.agent_search_config
        )
        self.extraction_provider = extraction_provider or VLLMExplicitFeatureExtractionProvider(
            config=config,
            output_dir=self.artifact_dir,
        )

        self.nuisance_rows: List[pd.DataFrame] = []
        self.r_stage_rows: List[pd.DataFrame] = []
        self.nuisance_attention_rows: List[Dict[str, Any]] = []
        self.effect_attention_rows: List[Dict[str, Any]] = []
        self.confounder_candidate_rows: List[Dict[str, Any]] = []
        self.modifier_candidate_rows: List[Dict[str, Any]] = []
        self.consensus_disambiguation_rows: List[Dict[str, Any]] = []
        self.consensus_recovery_rows: List[Dict[str, Any]] = []
        self.value_harmonization_rows: List[Dict[str, Any]] = []
        self.consensus_rows: List[Dict[str, Any]] = []
        self.coverage_filter_rows: List[Dict[str, Any]] = []
        self.association_filter_rows: List[Dict[str, Any]] = []
        self.metric_rows: List[Dict[str, Any]] = []

    @property
    def device(self) -> torch.device:
        return getattr(self._thread_state, "device", self._device)

    @device.setter
    def device(self, value: torch.device | str) -> None:
        self._device = torch.device(value)

    @staticmethod
    def _resolve_device_pool(
        *,
        base_device: torch.device,
        gpu_ids: Optional[Sequence[int]],
        devices: Optional[Sequence[torch.device | str]],
    ) -> List[torch.device]:
        if devices:
            return [torch.device(device) for device in devices]
        if gpu_ids and base_device.type == "cuda":
            return [torch.device(f"cuda:{int(gpu_id)}") for gpu_id in gpu_ids]
        return [base_device]

    @contextmanager
    def _using_device(self, device: torch.device | str):
        previous = getattr(self._thread_state, "device", None)
        self._thread_state.device = torch.device(device)
        try:
            yield
        finally:
            if previous is None:
                try:
                    delattr(self._thread_state, "device")
                except AttributeError:
                    pass
            else:
                self._thread_state.device = previous

    def run(self) -> None:
        logger.info("=" * 80)
        logger.info("AGENTIC ATTENTION VARIABLE FOREST")
        logger.info("=" * 80)

        splits = self._analysis_splits()
        outer_n_jobs = self._outer_n_jobs(len(splits))
        if outer_n_jobs > 1 and self._has_external_components:
            logger.warning(
                "Outer fold parallelism disabled because custom proposal_agent or "
                "extraction_provider objects were supplied and may not be thread-safe."
            )
            outer_n_jobs = 1

        if outer_n_jobs > 1:
            device_groups = self._outer_device_groups(outer_n_jobs)
            device_group_queue = self._outer_device_group_queue(outer_n_jobs)

            def run_with_device_group(
                outer_fold: int,
                train_idx: np.ndarray,
                test_idx: np.ndarray,
            ) -> Dict[str, Any]:
                devices = device_group_queue.get()
                try:
                    return self._run_one_analysis_split_isolated(
                        outer_fold=outer_fold,
                        train_idx=train_idx,
                        test_idx=test_idx,
                        devices=devices,
                        outer_n_jobs=outer_n_jobs,
                    )
                finally:
                    device_group_queue.put(devices)

            logger.info(
                "Running %s attention-variable outer fold(s) with "
                "outer_parallelism=%s device_groups=%s",
                len(splits),
                outer_n_jobs,
                [[str(device) for device in group] for group in device_groups],
            )
            with ThreadPoolExecutor(
                max_workers=outer_n_jobs,
                thread_name_prefix="avf-outer",
            ) as executor:
                futures = [
                    executor.submit(
                        run_with_device_group,
                        int(outer_fold),
                        np.asarray(train_idx),
                        np.asarray(test_idx),
                    )
                    for outer_fold, train_idx, test_idx in splits
                ]
                fold_results = [future.result() for future in futures]
            fold_results = sorted(fold_results, key=lambda item: item["outer_fold"])
            prediction_frames = [item["predictions"] for item in fold_results]
            for item in fold_results:
                self.nuisance_rows.extend(item["nuisance_rows"])
                self.r_stage_rows.extend(item["r_stage_rows"])
                self.nuisance_attention_rows.extend(item["nuisance_attention_rows"])
                self.effect_attention_rows.extend(item["effect_attention_rows"])
                self.confounder_candidate_rows.extend(item["confounder_candidate_rows"])
                self.modifier_candidate_rows.extend(item["modifier_candidate_rows"])
                self.consensus_disambiguation_rows.extend(item["consensus_disambiguation_rows"])
                self.consensus_recovery_rows.extend(item["consensus_recovery_rows"])
                self.value_harmonization_rows.extend(item["value_harmonization_rows"])
                self.consensus_rows.extend(item["consensus_rows"])
                self.coverage_filter_rows.extend(item["coverage_filter_rows"])
                self.association_filter_rows.extend(item["association_filter_rows"])
                self.metric_rows.extend(item["metric_rows"])
        else:
            prediction_frames = []
            for outer_fold, train_idx, test_idx in splits:
                logger.info(
                    "Attention-variable fold %s: train=%s test=%s device=%s",
                    outer_fold,
                    len(train_idx),
                    len(test_idx),
                    self.device,
                )
                fold_predictions = self._run_one_analysis_split(
                    outer_fold=outer_fold,
                    train_idx=train_idx,
                    test_idx=test_idx,
                )
                prediction_frames.append(fold_predictions)

        results_df = pd.concat(prediction_frames).sort_values("_oci_row_id")
        self._save_predictions(results_df)
        self._save_artifacts(results_df)

    def _run_one_analysis_split_isolated(
        self,
        outer_fold: int,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        devices: Sequence[torch.device],
        outer_n_jobs: int,
    ) -> Dict[str, Any]:
        assigned_devices = [torch.device(device) for device in devices]
        device = assigned_devices[0]
        logger.info(
            "Attention-variable isolated fold %s: train=%s test=%s devices=%s",
            outer_fold,
            len(train_idx),
            len(test_idx),
            [str(item) for item in assigned_devices],
        )
        fold_runner = AgenticAttentionVariableForestRunner(
            dataset=self.dataset,
            config=self.config,
            output_path=(
                self.artifact_dir / f"outer_fold_{int(outer_fold):03d}" / "predictions.parquet"
            ),
            device=device,
            num_workers=self._inner_workers_for_outer_job(outer_n_jobs),
            devices=assigned_devices,
        )
        predictions = fold_runner._run_one_analysis_split(
            outer_fold=outer_fold,
            train_idx=train_idx,
            test_idx=test_idx,
        )
        return {
            "outer_fold": int(outer_fold),
            "predictions": predictions,
            "nuisance_rows": fold_runner.nuisance_rows,
            "r_stage_rows": fold_runner.r_stage_rows,
            "nuisance_attention_rows": fold_runner.nuisance_attention_rows,
            "effect_attention_rows": fold_runner.effect_attention_rows,
            "confounder_candidate_rows": fold_runner.confounder_candidate_rows,
            "modifier_candidate_rows": fold_runner.modifier_candidate_rows,
            "consensus_disambiguation_rows": (fold_runner.consensus_disambiguation_rows),
            "consensus_recovery_rows": fold_runner.consensus_recovery_rows,
            "value_harmonization_rows": fold_runner.value_harmonization_rows,
            "consensus_rows": fold_runner.consensus_rows,
            "coverage_filter_rows": fold_runner.coverage_filter_rows,
            "association_filter_rows": fold_runner.association_filter_rows,
            "metric_rows": fold_runner.metric_rows,
        }

    def _analysis_splits(self) -> List[Tuple[int, np.ndarray, np.ndarray]]:
        return standalone_outer_splits(self.dataset, self.config)

    def _run_one_analysis_split(
        self,
        outer_fold: int,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
    ) -> pd.DataFrame:
        discovery_df = self.dataset.iloc[train_idx].reset_index(drop=True)
        nuisance = self._crossfit_nuisance(discovery_df, outer_fold)
        if getattr(self.avf_config, "neural_only", False):
            r_stage = self._crossfit_effect(discovery_df, nuisance["predictions"], outer_fold)
            predictions = self._neural_only_prediction_frame(
                discovery_df=discovery_df,
                r_stage_predictions=r_stage["predictions"],
                outer_fold=outer_fold,
            )
            metrics = self._neural_only_metrics(predictions)
            self.metric_rows.append(
                {
                    "outer_fold": outer_fold,
                    **metrics,
                }
            )
            return predictions

        confounders = self._discover_extract_filter_with_retries(
            stage="confounder",
            outer_fold=outer_fold,
            discovery_df=discovery_df,
            train_idx=train_idx,
            attention_rows=nuisance["attention"],
            existing_specs=self._initial_specs(),
        )
        r_stage = self._crossfit_effect(discovery_df, nuisance["predictions"], outer_fold)
        modifier_attention = r_stage["attention"]
        modifiers = self._discover_extract_filter_with_retries(
            stage="effect_modifier",
            outer_fold=outer_fold,
            discovery_df=discovery_df,
            train_idx=train_idx,
            attention_rows=modifier_attention,
            existing_specs=self._merge_specs(self._initial_specs(), confounders),
        )
        selected_specs = self._merge_specs(self._initial_specs(), confounders, modifiers)

        self.dataset = self.extraction_provider.ensure_features(self.dataset, selected_specs)
        train_df = self.dataset.iloc[train_idx].copy()
        test_df = self.dataset.iloc[test_idx].copy()
        selected_specs = self._filter_specs_by_extraction_coverage(
            train_df,
            selected_specs,
            manual_specs=self._initial_specs(),
        )
        self.consensus_rows.append(
            {
                "outer_fold": outer_fold,
                "selected_features": [_spec_to_dict(spec) for spec in selected_specs],
                "confounders": [spec.name for spec in selected_specs if "confounder" in spec.roles],
                "effect_modifiers": [
                    spec.name for spec in selected_specs if "effect_modifier" in spec.roles
                ],
            }
        )
        predictions, metrics, fit_audit = self._fit_final_forest(
            train_df=train_df,
            test_df=test_df,
            selected_specs=selected_specs,
            fold_id=outer_fold,
        )
        self.consensus_rows[-1]["final_causal_forest_fit_audit"] = fit_audit
        predictions["outer_fold"] = outer_fold
        predictions["selected_feature_names"] = ",".join(spec.name for spec in selected_specs)
        self.metric_rows.append({"outer_fold": outer_fold, **metrics})
        return predictions

    def _neural_only_prediction_frame(
        self,
        discovery_df: pd.DataFrame,
        r_stage_predictions: pd.DataFrame,
        outer_fold: int,
    ) -> pd.DataFrame:
        """Build a long-form prediction frame for neural-only diagnostics."""
        excluded_text_cols = {
            self.config.text_column,
            "patient_prompt",
            "event_timeline",
            "llm_raw_response",
        }
        metadata_cols = [
            col
            for col in discovery_df.columns
            if col == "_oci_row_id" or col not in excluded_text_cols
        ]
        metadata = discovery_df[metadata_cols].drop_duplicates("_oci_row_id")
        predictions = r_stage_predictions.merge(
            metadata,
            on="_oci_row_id",
            how="left",
            suffixes=("", "_source"),
        )
        predictions["outer_fold"] = int(outer_fold)
        predictions["pred_ite_prob"] = predictions["tau_hat_r_stage"]
        predictions["pred_propensity_prob"] = predictions["e_hat"]
        predictions["pred_outcome_prob"] = predictions["m_hat"]
        predictions["selected_feature_names"] = ""
        predictions["neural_only"] = True
        return predictions

    def _neural_only_metrics(self, predictions: pd.DataFrame) -> Dict[str, Any]:
        """Summarize nuisance and R-stage diagnostics for neural-only runs."""
        metrics: Dict[str, Any] = {
            "mode": "neural_only",
            "n_train_rows": int(len(predictions)),
            "effect_objective": str(
                predictions["effect_objective"].iloc[0]
                if "effect_objective" in predictions.columns and len(predictions) > 0
                else effect_objective_name(self.avf_config)
            ),
            "r_loss_mean": _finite_or_none(predictions["r_loss"].mean()),
            "r_loss_at_zero_tau_mean": _finite_or_none(predictions["r_loss_at_zero_tau"].mean()),
            "tau_hat_r_stage_mean": _finite_or_none(predictions["tau_hat_r_stage"].mean()),
            "tau_hat_r_stage_std": _finite_or_none(predictions["tau_hat_r_stage"].std()),
        }
        zero_loss = metrics.get("r_loss_at_zero_tau_mean")
        r_loss = metrics.get("r_loss_mean")
        if zero_loss is not None and zero_loss > 0 and r_loss is not None:
            metrics["r_loss_relative_improvement"] = float(1.0 - r_loss / zero_loss)
        if {
            "effect_loss",
            "effect_loss_at_zero_tau",
        }.issubset(predictions.columns):
            metrics["effect_loss_mean"] = _finite_or_none(predictions["effect_loss"].mean())
            metrics["effect_loss_at_zero_tau_mean"] = _finite_or_none(
                predictions["effect_loss_at_zero_tau"].mean()
            )
            effect_zero = metrics.get("effect_loss_at_zero_tau_mean")
            effect_loss = metrics.get("effect_loss_mean")
            if effect_zero is not None and effect_zero > 0 and effect_loss is not None:
                metrics["effect_loss_relative_improvement"] = float(1.0 - effect_loss / effect_zero)
        if "tau_logit_modifier" in predictions.columns:
            modifier = predictions["tau_logit_modifier"].to_numpy(dtype=float)
            finite = modifier[np.isfinite(modifier)]
            metrics["tau_logit_modifier_mean"] = (
                _finite_or_none(np.mean(finite)) if finite.size > 0 else None
            )
            metrics["tau_logit_modifier_std"] = (
                _finite_or_none(np.std(finite)) if finite.size > 0 else None
            )
        if "r_stage_train_eligible" in predictions.columns:
            eligible = predictions["r_stage_train_eligible"].astype(bool)
            metrics["r_stage_train_eligible_rows"] = int(eligible.sum())
            metrics["r_stage_train_eligible_fraction"] = _finite_or_none(eligible.mean())
        if self.config.treatment_column in predictions.columns:
            t = predictions[self.config.treatment_column].to_numpy()
            e_hat = predictions["e_hat"].to_numpy()
            metrics["nuisance_treatment_auroc"] = _safe_roc_auc(t, e_hat)
            metrics.update(binary_calibration_metrics(t, e_hat, prefix="nuisance_treatment"))
            if "e_hat_raw" in predictions.columns:
                e_raw = predictions["e_hat_raw"].to_numpy()
                metrics["nuisance_treatment_raw_auroc"] = _safe_roc_auc(t, e_raw)
                metrics.update(
                    binary_calibration_metrics(t, e_raw, prefix="nuisance_treatment_raw")
                )
        if self.config.outcome_column in predictions.columns:
            y = predictions[self.config.outcome_column].to_numpy()
            m_hat = predictions["m_hat"].to_numpy()
            if self.config.outcome_type == "continuous":
                metrics["nuisance_outcome_rmse"] = _finite_or_none(
                    np.sqrt(mean_squared_error(y, m_hat))
                )
            else:
                metrics["nuisance_outcome_auroc"] = _safe_roc_auc(y, m_hat)
                metrics.update(binary_calibration_metrics(y, m_hat, prefix="nuisance_outcome"))
                if "m_hat_raw" in predictions.columns:
                    m_raw = predictions["m_hat_raw"].to_numpy()
                    metrics["nuisance_outcome_raw_auroc"] = _safe_roc_auc(y, m_raw)
                    metrics.update(
                        binary_calibration_metrics(y, m_raw, prefix="nuisance_outcome_raw")
                    )
        return metrics

    def _initial_specs(self) -> List[ExplicitFeatureSpec]:
        if getattr(self.config.explicit_features, "features", None):
            return list(self.config.explicit_features.features)
        return []

    def _crossfit_nuisance(self, df: pd.DataFrame, outer_fold: int) -> Dict[str, Any]:
        folds = _bounded_fold_count(self.avf_config.nuisance_folds, len(df))
        predictions = pd.DataFrame(
            {
                "_oci_row_id": df["_oci_row_id"].to_numpy(),
                "outer_fold": outer_fold,
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
        attention_rows: List[Dict[str, Any]] = []

        split_items = list(
            enumerate(
                KFold(n_splits=folds, shuffle=True, random_state=10_000 + outer_fold).split(df),
                start=1,
            )
        )
        checkpoint_fingerprint = self._crossfit_checkpoint_fingerprint(
            "nuisance",
            folds,
            extra_payload={"nuisance_calibration_scope": NUISANCE_CALIBRATION_SCOPE},
        )

        def run_fold(fold: int, fit_pos: np.ndarray, heldout_pos: np.ndarray):
            cached = self._load_nuisance_fold_checkpoint(
                df=df,
                outer_fold=outer_fold,
                fold=fold,
                heldout_pos=heldout_pos,
                fingerprint=checkpoint_fingerprint,
            )
            if cached is not None:
                return cached

            model = None
            logger.info(
                "Outer fold %s nuisance fold %s/%s: train=%s heldout=%s%s",
                outer_fold,
                fold,
                folds,
                len(fit_pos),
                len(heldout_pos),
                self._cuda_memory_summary(),
            )
            try:
                model = NuisanceNet(
                    extractor=self._create_extractor(),
                    hidden_dim=getattr(
                        self.config.architecture,
                        "htr_prediction_head_hidden_dim",
                        64,
                    ),
                    outcome_type=self.config.outcome_type,
                ).to(self.device)
                self._train_nuisance_model(
                    model,
                    df,
                    fit_pos,
                    outer_fold=outer_fold,
                    fold=fold,
                    total_folds=folds,
                )
                heldout = df.iloc[heldout_pos]
                logger.info(
                    "Outer fold %s nuisance fold %s/%s: predicting heldout%s",
                    outer_fold,
                    fold,
                    folds,
                    self._cuda_memory_summary(),
                )
                # Only raw held-out predictions are produced here. Calibration
                # is cross-fitted after every fold has finished (see
                # _crossfit_calibrate_nuisance), so no calibrator ever sees the
                # in-sample predictions of the model it corrects.
                e_raw, m_raw = self._predict_nuisance_model(model, heldout)
                logger.info(
                    "Outer fold %s nuisance fold %s/%s: collecting attention evidence",
                    outer_fold,
                    fold,
                    folds,
                )
                fold_attention = self._attention_evidence(
                    model.extractor,
                    heldout,
                    fold=fold,
                    outer_fold=outer_fold,
                    stage="nuisance",
                    extra={"e_hat_raw": e_raw, "m_hat_raw": m_raw},
                )
                logger.info(
                    "Outer fold %s nuisance fold %s/%s complete: attention_rows=%s "
                    "e_hat_raw_mean=%.4f m_hat_raw_mean=%.4f%s",
                    outer_fold,
                    fold,
                    folds,
                    len(fold_attention),
                    float(np.mean(e_raw)),
                    float(np.mean(m_raw)),
                    self._cuda_memory_summary(),
                )
                result = {
                    "fold": fold,
                    "heldout_pos": heldout_pos,
                    "e_hat_raw": e_raw,
                    "m_hat_raw": m_raw,
                    "attention": fold_attention,
                }
                self._save_nuisance_fold_checkpoint(
                    df=df,
                    result=result,
                    outer_fold=outer_fold,
                    fingerprint=checkpoint_fingerprint,
                )
                return result
            except RuntimeError as exc:
                if _is_cuda_oom(exc):
                    logger.error(
                        "CUDA OOM in outer fold %s nuisance fold %s/%s%s",
                        outer_fold,
                        fold,
                        folds,
                        self._cuda_memory_summary(),
                    )
                raise
            finally:
                if model is not None:
                    self._cleanup_model(model)
                    model = None
                    logger.info(
                        "Outer fold %s nuisance fold %s/%s: model cleanup complete%s",
                        outer_fold,
                        fold,
                        folds,
                        self._cuda_memory_summary(),
                    )

        n_jobs = self._fold_n_jobs(folds)
        logger.info(
            "Outer fold %s nuisance cross-fit parallelism: folds=%s n_jobs=%s "
            "setting=%s device=%s",
            outer_fold,
            folds,
            n_jobs,
            self.avf_config.fold_parallelism,
            self.device,
        )
        fold_results = _run_crossfit_fold_tasks(
            run_fold,
            split_items,
            n_jobs,
            device_context=self._device_context_for_inner_fold,
        )
        self._crossfit_calibrate_nuisance(df, fold_results, outer_fold=outer_fold)

        for result in fold_results:
            heldout_pos = result["heldout_pos"]
            predictions.loc[heldout_pos, "e_hat"] = result["e_hat"]
            predictions.loc[heldout_pos, "e_hat_raw"] = result["e_hat_raw"]
            predictions.loc[heldout_pos, "m_hat"] = result["m_hat"]
            predictions.loc[heldout_pos, "m_hat_raw"] = result["m_hat_raw"]
            predictions.loc[heldout_pos, "y_residual"] = result["y_resid"]
            predictions.loc[heldout_pos, "t_residual"] = result["t_resid"]
            predictions.loc[heldout_pos, "r_loss_at_zero_tau"] = result["y_resid"] ** 2
            predictions.loc[heldout_pos, "nuisance_fold"] = result["fold"]
            attention_rows.extend(result["attention"])

        self.nuisance_rows.append(predictions)
        self.nuisance_attention_rows.extend(attention_rows)
        return {"predictions": predictions, "attention": attention_rows}

    def _crossfit_calibrate_nuisance(
        self,
        df: pd.DataFrame,
        fold_results: Sequence[Dict[str, Any]],
        *,
        outer_fold: int,
    ) -> None:
        """Cross-fitted nuisance calibration (in place on ``fold_results``).

        Uses the shared Stage 1 helper ``cross_fitted_calibrators``: the
        calibrator applied to fold k is fit on the pooled raw held-out
        predictions and labels of the other folds. Those rows are all inside
        fold k's training rows, and each prediction comes from a model that
        did not train on that row, so the calibrator learns the networks'
        out-of-sample miscalibration rather than in-sample overfit.
        """

        t_all = df[self.config.treatment_column].to_numpy(dtype=float)
        y_all = df[self.config.outcome_column].to_numpy(dtype=float)
        row_ids = df["_oci_row_id"].to_numpy()
        method = self.avf_config.nuisance_calibration
        total = len(fold_results)
        positions = np.concatenate(
            [np.asarray(result["heldout_pos"], dtype=int) for result in fold_results]
        )
        fold_ids = np.concatenate(
            [np.full(len(result["heldout_pos"]), int(result["fold"])) for result in fold_results]
        )
        e_calibrators = cross_fitted_calibrators(
            np.concatenate([np.asarray(r["e_hat_raw"], dtype=float) for r in fold_results]),
            t_all[positions],
            fold_ids,
            method=method,
        )
        m_calibrators = (
            None
            if self.config.outcome_type == "continuous"
            else cross_fitted_calibrators(
                np.concatenate([np.asarray(r["m_hat_raw"], dtype=float) for r in fold_results]),
                y_all[positions],
                fold_ids,
                method=method,
            )
        )
        for result in fold_results:
            heldout_pos = np.asarray(result["heldout_pos"], dtype=int)
            e_raw = np.asarray(result["e_hat_raw"], dtype=float)
            m_raw = np.asarray(result["m_hat_raw"], dtype=float)
            e_hat = e_calibrators[int(result["fold"])].transform(e_raw)
            m_hat = (
                m_raw
                if m_calibrators is None
                else m_calibrators[int(result["fold"])].transform(m_raw)
            )
            y = y_all[heldout_pos]
            t = t_all[heldout_pos]
            y_resid = y - m_hat
            t_resid = t - e_hat
            result.update(e_hat=e_hat, m_hat=m_hat, y_resid=y_resid, t_resid=t_resid)
            calibrated = {
                row_id: (float(e), float(m), float(yr), float(tr))
                for row_id, e, m, yr, tr in zip(
                    row_ids[heldout_pos], e_hat, m_hat, y_resid, t_resid
                )
            }
            for record in result["attention"]:
                values = calibrated.get(record.get("row_id"))
                if values is not None:
                    record.update(
                        e_hat=values[0],
                        m_hat=values[1],
                        y_residual=values[2],
                        t_residual=values[3],
                    )
            binary = self.config.outcome_type != "continuous"
            logger.info(
                "Outer fold %s nuisance fold %s/%s heldout metrics (cross-fitted "
                "calibration): propensity_auroc=%s raw=%s propensity_ece=%s raw_ece=%s "
                "outcome_auroc=%s raw=%s outcome_ece=%s raw_ece=%s",
                outer_fold,
                result["fold"],
                total,
                _format_optional_metric(_safe_roc_auc(t, e_hat)),
                _format_optional_metric(_safe_roc_auc(t, e_raw)),
                _format_optional_metric(binary_calibration_metrics(t, e_hat)["ece"]),
                _format_optional_metric(binary_calibration_metrics(t, e_raw)["ece"]),
                _format_optional_metric(_safe_roc_auc(y, m_hat) if binary else None),
                _format_optional_metric(_safe_roc_auc(y, m_raw) if binary else None),
                _format_optional_metric(
                    binary_calibration_metrics(y, m_hat)["ece"] if binary else None
                ),
                _format_optional_metric(
                    binary_calibration_metrics(y, m_raw)["ece"] if binary else None
                ),
            )

    def _crossfit_effect(
        self,
        df: pd.DataFrame,
        nuisance_predictions: pd.DataFrame,
        outer_fold: int,
    ) -> Dict[str, Any]:
        folds = _bounded_fold_count(self.avf_config.effect_folds, len(df))
        effect_objective = effect_objective_name(self.avf_config)
        if effect_objective == "dina" and self.config.outcome_type != "binary":
            raise ValueError("Bernoulli effect objectives require binary outcomes")
        r_df = nuisance_predictions.copy()
        r_df["tau_hat_r_stage"] = np.nan
        r_df["tau_logit_modifier"] = np.nan
        r_df["r_loss"] = np.nan
        r_df["effect_loss"] = np.nan
        r_df["effect_loss_at_zero_tau"] = np.nan
        r_df["effect_fold"] = np.nan
        attention_rows: List[Dict[str, Any]] = []

        e = r_df["e_hat"].to_numpy(dtype=float)
        m = r_df["m_hat"].to_numpy(dtype=float)
        y = df[self.config.outcome_column].to_numpy(dtype=float)
        t = df[self.config.treatment_column].to_numpy(dtype=float)
        r_stage_min_propensity = float(getattr(self.avf_config, "r_stage_min_propensity", 0.0))
        r_stage_max_propensity = float(getattr(self.avf_config, "r_stage_max_propensity", 1.0))
        train_eligible = (
            np.isfinite(e) & (e >= r_stage_min_propensity) & (e <= r_stage_max_propensity)
        )
        e_clipped = np.clip(e, self.avf_config.e_clip, 1.0 - self.avf_config.e_clip)
        m_clipped = clip_probability(m)
        t_resid = t - e_clipped
        y_resid = y - m
        r_pseudo_outcome = _r_pseudo_outcome(y_resid, t_resid)
        if effect_objective == "pseudo_outcome_mse":
            train_eligible = train_eligible & np.isfinite(r_pseudo_outcome)
        r_df["r_stage_train_eligible"] = train_eligible
        r_df["effect_objective"] = effect_objective
        r_df["r_pseudo_outcome"] = r_pseudo_outcome
        if effect_objective == "squared_r_loss":
            r_df["effect_loss_at_zero_tau"] = y_resid**2
        else:
            r_df["effect_loss_at_zero_tau"] = r_pseudo_outcome**2

        split_items = list(
            enumerate(
                KFold(n_splits=folds, shuffle=True, random_state=20_000 + outer_fold).split(df),
                start=1,
            )
        )
        checkpoint_fingerprint = self._crossfit_checkpoint_fingerprint(
            "r_stage",
            folds,
            extra_payload={
                "e_hat_hash": _hash_numeric_array(e),
                "m_hat_hash": _hash_numeric_array(m),
                "effect_objective": effect_objective,
                "effect_loss": f"{effect_objective}_v1",
                "r_stage_min_propensity": r_stage_min_propensity,
                "r_stage_max_propensity": r_stage_max_propensity,
            },
        )

        def run_fold(fold: int, fit_pos: np.ndarray, heldout_pos: np.ndarray):
            fit_pos = np.asarray(fit_pos, dtype=int)
            heldout_pos = np.asarray(heldout_pos, dtype=int)
            eligible_fit_pos = fit_pos[train_eligible[fit_pos]]
            cached = self._load_effect_fold_checkpoint(
                df=df,
                outer_fold=outer_fold,
                fold=fold,
                heldout_pos=heldout_pos,
                fingerprint=checkpoint_fingerprint,
            )
            if cached is not None:
                return cached

            if len(eligible_fit_pos) < 1:
                raise ValueError(
                    "No rows remain for R-stage training in outer fold "
                    f"{outer_fold} effect fold {fold} after applying propensity "
                    f"bounds [{r_stage_min_propensity}, {r_stage_max_propensity}]"
                )

            model = None
            logger.info(
                "Outer fold %s effect fold %s/%s: train=%s/%s eligible "
                "heldout=%s propensity_bounds=[%.3f, %.3f]%s",
                outer_fold,
                fold,
                folds,
                len(eligible_fit_pos),
                len(fit_pos),
                len(heldout_pos),
                r_stage_min_propensity,
                r_stage_max_propensity,
                self._cuda_memory_summary(),
            )
            try:
                model = EffectNet(
                    extractor=self._create_extractor(),
                    hidden_dim=getattr(
                        self.config.architecture,
                        "htr_prediction_head_hidden_dim",
                        64,
                    ),
                ).to(self.device)
                self._train_effect_model(
                    model,
                    df,
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
                heldout = df.iloc[heldout_pos]
                logger.info(
                    "Outer fold %s effect fold %s/%s: predicting heldout%s",
                    outer_fold,
                    fold,
                    folds,
                    self._cuda_memory_summary(),
                )
                raw_effect = self._predict_effect_model(model, heldout)
                heldout_pseudo_outcome = r_pseudo_outcome[heldout_pos]
                if effect_objective == "dina":
                    from .stage1_dina import neural_predictions
                    from ..models import dina

                    dp = neural_predictions(self, model, heldout, raw_effect)
                    tau_hat, tau_logit_modifier = dp["tau"], dp["delta"]
                    heldout_effect_loss = dina.loss(
                        y[heldout_pos], t[heldout_pos], dp["a"], dp["nu"], dp["delta"]
                    )
                    r_df.loc[heldout_pos, "effect_loss_at_zero_tau"] = dina.loss(
                        y[heldout_pos], t[heldout_pos], dp["a"], dp["nu"], 0.0
                    )
                else:
                    tau_hat = raw_effect
                    tau_logit_modifier = np.full(len(heldout_pos), np.nan)
                    if effect_objective == "pseudo_outcome_mse":
                        heldout_effect_loss = (tau_hat - heldout_pseudo_outcome) ** 2
                    else:
                        heldout_effect_loss = (
                            y_resid[heldout_pos] - tau_hat * t_resid[heldout_pos]
                        ) ** 2
                heldout_r_loss = (y_resid[heldout_pos] - tau_hat * t_resid[heldout_pos]) ** 2
                logger.info(
                    "Outer fold %s effect fold %s/%s: collecting attention evidence",
                    outer_fold,
                    fold,
                    folds,
                )
                fold_attention = self._attention_evidence(
                    model.extractor,
                    heldout,
                    fold=fold,
                    outer_fold=outer_fold,
                    stage="effect_modifier",
                    extra={
                        "tau_hat_r_stage": tau_hat,
                        "tau_logit_modifier": tau_logit_modifier,
                        "r_pseudo_outcome": heldout_pseudo_outcome,
                        "r_loss": heldout_r_loss,
                        "effect_loss": heldout_effect_loss,
                        "effect_objective": np.asarray(
                            [effect_objective] * len(heldout_pos),
                            dtype=object,
                        ),
                    },
                )
                logger.info(
                    "Outer fold %s effect fold %s/%s complete: attention_rows=%s "
                    "objective=%s tau_mean=%.4f r_loss_mean=%.4f "
                    "effect_loss_mean=%.4f%s",
                    outer_fold,
                    fold,
                    folds,
                    len(fold_attention),
                    effect_objective,
                    float(np.mean(tau_hat)),
                    float(np.mean(heldout_r_loss)),
                    float(np.mean(heldout_effect_loss)),
                    self._cuda_memory_summary(),
                )
                result = {
                    "fold": fold,
                    "heldout_pos": heldout_pos,
                    "tau_hat": tau_hat,
                    "tau_logit_modifier": tau_logit_modifier,
                    "r_pseudo_outcome": heldout_pseudo_outcome,
                    "r_loss": heldout_r_loss,
                    "effect_loss": heldout_effect_loss,
                    "effect_loss_at_zero_tau": r_df.iloc[heldout_pos][
                        "effect_loss_at_zero_tau"
                    ].to_numpy(dtype=float),
                    "effect_objective": effect_objective,
                    "attention": fold_attention,
                    "r_stage_train_eligible": train_eligible[heldout_pos],
                }
                self._save_effect_fold_checkpoint(
                    df=df,
                    result=result,
                    outer_fold=outer_fold,
                    fingerprint=checkpoint_fingerprint,
                )
                return result
            except RuntimeError as exc:
                if _is_cuda_oom(exc):
                    logger.error(
                        "CUDA OOM in outer fold %s effect fold %s/%s%s",
                        outer_fold,
                        fold,
                        folds,
                        self._cuda_memory_summary(),
                    )
                raise
            finally:
                if model is not None:
                    self._cleanup_model(model)
                    model = None
                    logger.info(
                        "Outer fold %s effect fold %s/%s: model cleanup complete%s",
                        outer_fold,
                        fold,
                        folds,
                        self._cuda_memory_summary(),
                    )

        n_jobs = self._fold_n_jobs(folds)
        logger.info(
            "Outer fold %s effect cross-fit parallelism: folds=%s n_jobs=%s "
            "setting=%s device=%s",
            outer_fold,
            folds,
            n_jobs,
            self.avf_config.fold_parallelism,
            self.device,
        )
        fold_results = _run_crossfit_fold_tasks(
            run_fold,
            split_items,
            n_jobs,
            device_context=self._device_context_for_inner_fold,
        )

        for result in fold_results:
            heldout_pos = result["heldout_pos"]
            r_df.loc[heldout_pos, "tau_hat_r_stage"] = result["tau_hat"]
            r_df.loc[heldout_pos, "tau_logit_modifier"] = result.get(
                "tau_logit_modifier",
                np.full(len(heldout_pos), np.nan),
            )
            r_df.loc[heldout_pos, "r_loss"] = result["r_loss"]
            r_df.loc[heldout_pos, "effect_loss"] = result.get(
                "effect_loss",
                result["r_loss"],
            )
            r_df.loc[heldout_pos, "effect_loss_at_zero_tau"] = result.get(
                "effect_loss_at_zero_tau",
                r_df.iloc[heldout_pos]["r_loss_at_zero_tau"].to_numpy(dtype=float),
            )
            r_df.loc[heldout_pos, "effect_fold"] = result["fold"]
            attention_rows.extend(result["attention"])

        self.r_stage_rows.append(r_df)
        self.effect_attention_rows.extend(attention_rows)
        return {"predictions": r_df, "attention": attention_rows}

    def _discover_variables_from_attention(
        self,
        stage: str,
        outer_fold: int,
        discovery_df: pd.DataFrame,
        attention_rows: Sequence[Dict[str, Any]],
        existing_specs: Sequence[ExplicitFeatureSpec],
        proposal_attempt: int = 1,
        rejected_low_coverage: Optional[Sequence[Dict[str, Any]]] = None,
        rejected_low_signal: Optional[Sequence[Dict[str, Any]]] = None,
        multivariable_signal_feedback: Optional[Dict[str, Any]] = None,
        excluded_feature_names: Optional[Sequence[str]] = None,
    ) -> List[ExplicitFeatureSpec]:
        proposals_by_fold: Dict[int, List[ExplicitFeatureSpec]] = {}
        proposal_artifacts_by_fold: Dict[int, List[Dict[str, Any]]] = {}
        proposal_limit = self._candidate_proposal_limit()
        excluded_names = {
            _normalize_feature_name(name)
            for name in (excluded_feature_names or [])
            if _normalize_feature_name(name)
        }
        fold_ids = sorted(
            {int(row["fold"]) for row in attention_rows if row.get("fold") is not None}
        )
        candidate_n_jobs = self._agent_candidate_n_jobs(len(fold_ids))

        def propose_for_fold(fold: int) -> Dict[str, Any]:
            fold_rows = [row for row in attention_rows if int(row.get("fold")) == fold]
            context = self._build_agent_context(
                stage=stage,
                outer_fold=outer_fold,
                inner_fold=fold,
                discovery_df=discovery_df,
                attention_rows=fold_rows,
                existing_specs=existing_specs,
                proposal_attempt=proposal_attempt,
                max_proposals=proposal_limit,
                rejected_low_coverage=rejected_low_coverage or [],
                rejected_low_signal=rejected_low_signal or [],
                multivariable_signal_feedback=multivariable_signal_feedback or {},
                excluded_feature_names=sorted(excluded_names),
            )
            self._save_agent_candidate_checkpoint(
                {
                    "outer_fold": outer_fold,
                    "fold": fold,
                    "stage": stage,
                    "proposal_attempt": int(proposal_attempt),
                    "status": "started",
                    "context": self._stored_agent_context(context),
                },
                stage=stage,
                outer_fold=outer_fold,
                fold=fold,
            )
            proposal_agent = (
                OpenAICompatibleFeatureSearchAgent(self.agent_search_config)
                if candidate_n_jobs > 1
                else self.proposal_agent
            )
            try:
                raw = proposal_agent.propose(context)
            except Exception as exc:
                error_row = {
                    "outer_fold": outer_fold,
                    "fold": fold,
                    "stage": stage,
                    "proposal_attempt": int(proposal_attempt),
                    "status": "error",
                    "context": self._stored_agent_context(context),
                    "error": str(exc),
                }
                if getattr(self.agent_search_config, "save_agent_raw_output", False):
                    error_row["agent_raw_output"] = _get_agent_response_trace(proposal_agent)
                self._save_agent_candidate_checkpoint(
                    error_row,
                    stage=stage,
                    outer_fold=outer_fold,
                    fold=fold,
                )
                raise
            raw_proposals = _proposal_list(raw)
            specs = _proposal_dicts_to_specs(
                raw_proposals,
                required_role=stage,
                max_specs=proposal_limit,
                excluded_feature_names=excluded_names,
            )
            proposal_artifacts = _proposal_artifact_dicts(raw_proposals, specs)
            row = {
                "outer_fold": outer_fold,
                "fold": fold,
                "stage": stage,
                "proposal_attempt": int(proposal_attempt),
                "status": "complete",
                "context": self._stored_agent_context(context),
                "proposals": proposal_artifacts,
            }
            if getattr(self.agent_search_config, "save_agent_raw_output", False):
                row["agent_raw_output"] = _get_agent_response_trace(proposal_agent)
            return {
                "fold": fold,
                "specs": specs,
                "proposal_artifacts": proposal_artifacts,
                "row": row,
            }

        if candidate_n_jobs > 1:
            logger.info(
                "Outer fold %s %s proposal attempt %s: running %s inner-fold "
                "agent proposal calls with candidate_proposal_parallelism=%s",
                outer_fold,
                stage,
                proposal_attempt,
                candidate_n_jobs,
                self.avf_config.candidate_proposal_parallelism,
            )
            with ThreadPoolExecutor(
                max_workers=candidate_n_jobs,
                thread_name_prefix="avf-agent-candidate",
            ) as executor:
                futures = [executor.submit(propose_for_fold, fold) for fold in fold_ids]
                proposal_results = [future.result() for future in futures]
        else:
            proposal_results = [propose_for_fold(fold) for fold in fold_ids]

        for result in sorted(proposal_results, key=lambda item: int(item["fold"])):
            fold = int(result["fold"])
            proposals_by_fold[fold] = result["specs"]
            proposal_artifacts_by_fold[fold] = result["proposal_artifacts"]
            row = result["row"]
            if stage == "confounder":
                self.confounder_candidate_rows.append(row)
            else:
                self.modifier_candidate_rows.append(row)
            self._save_agent_candidate_checkpoint(
                row,
                stage=stage,
                outer_fold=outer_fold,
                fold=fold,
            )
            self._flush_agent_candidate_rows()

        disambiguation = self._resolve_consensus_disambiguation(
            stage=stage,
            outer_fold=outer_fold,
            proposal_attempt=proposal_attempt,
            proposals_by_fold=proposals_by_fold,
            proposal_artifacts_by_fold=proposal_artifacts_by_fold,
            excluded_feature_names=sorted(excluded_names),
        )
        selected = consensus_feature_specs(
            proposals_by_fold,
            min_fold_fraction=self.avf_config.consensus_min_fold_fraction,
            required_role=stage,
            min_folds=getattr(self.avf_config, "consensus_min_folds", None),
            concept_groups=(
                disambiguation.get("validated_groups")
                if disambiguation.get("status") == "complete"
                else None
            ),
        )
        selected, recovery = self._resolve_consensus_recovery(
            stage=stage,
            outer_fold=outer_fold,
            proposal_attempt=proposal_attempt,
            proposals_by_fold=proposals_by_fold,
            proposal_artifacts_by_fold=proposal_artifacts_by_fold,
            disambiguation=disambiguation,
            fallback_selected=selected,
        )
        self.consensus_recovery_rows.append(recovery)
        self._flush_consensus_recovery_rows()
        selected_group_names = {spec.name for spec in selected}
        disambiguation["selected_groups"] = [
            group
            for group in disambiguation.get("validated_groups", [])
            if _normalize_feature_name(group.get("canonical_name")) in selected_group_names
        ]
        self.consensus_disambiguation_rows.append(disambiguation)
        self._flush_consensus_disambiguation_rows()
        logger.info(
            "Outer fold %s %s consensus selected %s variable(s): %s",
            outer_fold,
            stage,
            len(selected),
            [spec.name for spec in selected],
        )
        return selected

    def _build_consensus_candidate_summaries(
        self,
        *,
        stage: str,
        proposals_by_fold: Dict[int, List[ExplicitFeatureSpec]],
        proposal_artifacts_by_fold: Dict[int, List[Dict[str, Any]]],
        disambiguation: Dict[str, Any],
        threshold: int,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, ExplicitFeatureSpec]]:
        fold_ids = sorted(int(fold) for fold in proposals_by_fold)
        spec_by_fold_name: Dict[Tuple[int, str], ExplicitFeatureSpec] = {}
        artifact_by_fold_name: Dict[Tuple[int, str], Dict[str, Any]] = {}
        for fold, specs in proposals_by_fold.items():
            for spec in specs:
                if stage not in spec.roles:
                    continue
                name = _normalize_feature_name(spec.name)
                if name:
                    spec_by_fold_name[(int(fold), name)] = spec
        for fold, artifacts in proposal_artifacts_by_fold.items():
            for artifact in artifacts:
                name = _normalize_feature_name(artifact.get("name"))
                if name:
                    artifact_by_fold_name[(int(fold), name)] = artifact

        summaries: List[Dict[str, Any]] = []
        canonical_specs: Dict[str, ExplicitFeatureSpec] = {}
        used_pairs: set[Tuple[int, str]] = set()

        def add_summary(
            *,
            name: str,
            spec: ExplicitFeatureSpec,
            members: Sequence[Tuple[int, str, ExplicitFeatureSpec]],
            group_rationale: Optional[str] = None,
        ) -> None:
            normalized_name = _normalize_feature_name(name)
            if not normalized_name or not members:
                return
            support_folds = sorted({int(fold) for fold, _, _ in members})
            source_names = sorted({str(source_name) for _, source_name, _ in members})
            rationales: List[Dict[str, Any]] = []
            expected_signals: List[str] = []
            for fold, source_name, _ in members:
                artifact = artifact_by_fold_name.get((int(fold), source_name), {})
                rationale = artifact.get("rationale")
                if rationale:
                    rationales.append(
                        {
                            "fold": int(fold),
                            "name": source_name,
                            "text": str(rationale),
                        }
                    )
                expected_signal = artifact.get("expected_signal")
                if expected_signal:
                    expected_signals.append(str(expected_signal))
            if group_rationale:
                rationales.append(
                    {
                        "fold": None,
                        "name": normalized_name,
                        "text": str(group_rationale),
                    }
                )
            support_count = len(support_folds)
            support_fraction = float(support_count / len(fold_ids)) if fold_ids else None
            summary = {
                "name": normalized_name,
                "type": spec.type,
                "categories": spec.categories,
                "roles": list(spec.roles),
                "description": spec.description,
                "source_names": source_names,
                "support_folds": support_folds,
                "missing_folds": [int(fold) for fold in fold_ids if int(fold) not in support_folds],
                "support_count": int(support_count),
                "support_fraction": support_fraction,
                "passes_consensus_gate": bool(support_count >= threshold),
                "rationales": rationales[:5],
                "expected_signals": list(dict.fromkeys(expected_signals))[:5],
            }
            summaries.append(summary)
            canonical_specs[normalized_name] = spec

        if disambiguation.get("status") == "complete":
            for group in disambiguation.get("validated_groups", []):
                canonical_name = _normalize_feature_name(group.get("canonical_name"))
                group_members: List[Tuple[int, str, ExplicitFeatureSpec]] = []
                for member in group.get("members", []):
                    fold = int(member.get("fold"))
                    member_name = _normalize_feature_name(member.get("name"))
                    spec = spec_by_fold_name.get((fold, member_name))
                    if spec is None:
                        continue
                    group_members.append((fold, member_name, spec))
                    used_pairs.add((fold, member_name))
                if not canonical_name or not group_members:
                    continue
                group_type = str(group.get("type") or group_members[0][2].type).lower()
                if group_type not in VALID_TYPES:
                    group_type = group_members[0][2].type
                categories = group.get("categories") if group_type == "categorical" else None
                if categories is not None:
                    categories = [str(category) for category in categories[:8]]
                roles = [role for role in group.get("roles", []) if role in VALID_ROLES]
                if stage not in roles:
                    roles.append(stage)
                try:
                    group_spec = ExplicitFeatureSpec(
                        name=canonical_name,
                        type=group_type,
                        categories=categories,
                        description=(
                            group.get("description")
                            or group_members[0][2].description
                            or canonical_name.replace("_", " ")
                        ),
                        roles=roles,
                    )
                except ValueError:
                    continue
                add_summary(
                    name=canonical_name,
                    spec=group_spec,
                    members=group_members,
                    group_rationale=group.get("rationale"),
                )

        for fold in fold_ids:
            seen_in_fold: set[str] = set()
            for spec in proposals_by_fold.get(fold, []):
                if stage not in spec.roles:
                    continue
                name = _normalize_feature_name(spec.name)
                if not name or name in seen_in_fold:
                    continue
                seen_in_fold.add(name)
                if (int(fold), name) in used_pairs:
                    continue
                members = [
                    (int(member_fold), name, member_spec)
                    for (member_fold, member_name), member_spec in spec_by_fold_name.items()
                    if member_name == name
                ]
                for member_fold, member_name, _ in members:
                    used_pairs.add((int(member_fold), member_name))
                add_summary(name=name, spec=spec, members=members)

        summaries = self._rank_consensus_candidate_summaries(summaries)
        return summaries, canonical_specs

    @staticmethod
    def _rank_consensus_candidate_summaries(
        summaries: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        return sorted(
            [dict(summary) for summary in summaries],
            key=lambda item: (
                not bool(item.get("passes_consensus_gate")),
                -int(item.get("support_count", 0) or 0),
                str(item.get("name", "")),
            ),
        )

    def _build_consensus_recovery_context(
        self,
        *,
        stage: str,
        outer_fold: int,
        proposal_attempt: int,
        candidate_summaries: Sequence[Dict[str, Any]],
        threshold: int,
        fold_count: int,
    ) -> Dict[str, Any]:
        ranked = self._rank_consensus_candidate_summaries(candidate_summaries)
        passed = [item for item in ranked if item.get("passes_consensus_gate")]
        recovery_limit = int(getattr(self.avf_config, "consensus_recovery_max_candidates", 12))
        below_threshold = [item for item in ranked if not item.get("passes_consensus_gate")][
            :recovery_limit
        ]
        return {
            "prompt_version": "agentic_attention_consensus_recovery_v1",
            "stage": stage,
            "required_role": stage,
            "outer_fold": int(outer_fold),
            "proposal_attempt": int(proposal_attempt),
            "max_selected_candidates": self._candidate_proposal_limit(),
            "fold_count": int(fold_count),
            "min_support_folds": int(threshold),
            "min_support_fraction": float(self.avf_config.consensus_min_fold_fraction),
            "selection_policy": [
                "Keep candidates that pass the fold-consensus gate unless they are redundant or likely leakage.",
                "Recover below-threshold candidates only when fold absence appears unstable rather than truly absent.",
                "Do not invent variables outside candidate_summaries.",
            ],
            "candidate_summaries": passed + below_threshold,
        }

    def _resolve_consensus_recovery(
        self,
        *,
        stage: str,
        outer_fold: int,
        proposal_attempt: int,
        proposals_by_fold: Dict[int, List[ExplicitFeatureSpec]],
        proposal_artifacts_by_fold: Dict[int, List[Dict[str, Any]]],
        disambiguation: Dict[str, Any],
        fallback_selected: List[ExplicitFeatureSpec],
    ) -> Tuple[List[ExplicitFeatureSpec], Dict[str, Any]]:
        fold_count = len(proposals_by_fold)
        threshold = _consensus_threshold(
            fold_count=fold_count,
            min_fold_fraction=self.avf_config.consensus_min_fold_fraction,
            min_folds=getattr(self.avf_config, "consensus_min_folds", None),
        )
        candidate_summaries, canonical_specs = self._build_consensus_candidate_summaries(
            stage=stage,
            proposals_by_fold=proposals_by_fold,
            proposal_artifacts_by_fold=proposal_artifacts_by_fold,
            disambiguation=disambiguation,
            threshold=threshold,
        )
        base_row: Dict[str, Any] = {
            "outer_fold": int(outer_fold),
            "stage": stage,
            "proposal_attempt": int(proposal_attempt),
            "threshold": int(threshold),
            "fold_count": int(fold_count),
            "candidate_summaries": candidate_summaries,
            "fallback_selected_features": [_spec_to_dict(spec) for spec in fallback_selected],
            "selected_features": [_spec_to_dict(spec) for spec in fallback_selected],
            "rejected_proposals": [],
            "used_fallback": True,
        }
        if not bool(getattr(self.avf_config, "consensus_recovery_enabled", True)):
            return fallback_selected, {**base_row, "status": "skipped_disabled"}
        if not candidate_summaries:
            return fallback_selected, {**base_row, "status": "skipped_no_candidates"}

        context = self._build_consensus_recovery_context(
            stage=stage,
            outer_fold=outer_fold,
            proposal_attempt=proposal_attempt,
            candidate_summaries=candidate_summaries,
            threshold=threshold,
            fold_count=fold_count,
        )
        context_names = {
            str(item.get("name"))
            for item in context.get("candidate_summaries", [])
            if item.get("name")
        }
        if not context_names:
            return fallback_selected, {**base_row, "status": "skipped_no_context"}

        started_row = {
            **base_row,
            "status": "started",
            "context": self._stored_agent_context(context),
        }
        self._save_agent_consensus_recovery_checkpoint(
            started_row,
            stage=stage,
            outer_fold=outer_fold,
            proposal_attempt=proposal_attempt,
        )

        try:
            raw_selection = self.proposal_agent.propose(context)
            selection_trace = _get_agent_response_trace(self.proposal_agent)
            proposals, rejected = validate_agentic_proposals(
                _proposal_list(raw_selection),
                current_specs=[],
                search_config=self.agent_search_config,
                allow_removals=False,
                max_additions=self._candidate_proposal_limit(),
                allow_duplicate_additions=True,
            )
        except Exception as exc:
            logger.warning(
                "Attention consensus recovery failed; using consensus fallback",
                exc_info=True,
            )
            row = {
                **base_row,
                "status": "agent_error_fallback",
                "context": self._stored_agent_context(context),
                "error": str(exc),
            }
            self._save_agent_consensus_recovery_checkpoint(
                row,
                stage=stage,
                outer_fold=outer_fold,
                proposal_attempt=proposal_attempt,
            )
            return fallback_selected, row

        selected_specs: List[ExplicitFeatureSpec] = []
        seen_names: set[str] = set()
        for proposal in proposals:
            if proposal.action != "add":
                continue
            name = _normalize_feature_name(proposal.name)
            if name not in context_names or name not in canonical_specs:
                rejected.append(
                    {
                        "proposal": _proposal_to_dict(proposal),
                        "reason": "not_in_consensus_candidates",
                    }
                )
                continue
            if name in seen_names:
                continue
            seen_names.add(name)
            selected_specs.append(
                self._recovery_spec_from_proposal(
                    base_spec=canonical_specs[name],
                    proposal=proposal,
                    required_role=stage,
                )
            )

        used_fallback = False
        if not selected_specs:
            selected_specs = fallback_selected
            used_fallback = True

        row = {
            **base_row,
            "status": "complete",
            "context": self._stored_agent_context(context),
            "raw_selection": raw_selection,
            "valid_proposals": [_proposal_to_dict(proposal) for proposal in proposals],
            "rejected_proposals": rejected,
            "selected_features": [_spec_to_dict(spec) for spec in selected_specs],
            "used_fallback": used_fallback,
        }
        if getattr(self.agent_search_config, "save_agent_raw_output", False):
            row["agent_raw_output"] = selection_trace
        self._save_agent_consensus_recovery_checkpoint(
            row,
            stage=stage,
            outer_fold=outer_fold,
            proposal_attempt=proposal_attempt,
        )
        return selected_specs, row

    @staticmethod
    def _recovery_spec_from_proposal(
        *,
        base_spec: ExplicitFeatureSpec,
        proposal: AgenticFeatureProposal,
        required_role: str,
    ) -> ExplicitFeatureSpec:
        roles = [role for role in proposal.roles if role in VALID_ROLES]
        if not roles:
            roles = list(base_spec.roles)
        if required_role not in roles:
            roles.append(required_role)
        return ExplicitFeatureSpec(
            name=base_spec.name,
            type=base_spec.type,
            categories=base_spec.categories,
            description=proposal.description or base_spec.description,
            value_aliases=getattr(base_spec, "value_aliases", None),
            roles=list(dict.fromkeys(roles)),
        )

    def _discover_extract_filter_with_retries(
        self,
        stage: str,
        outer_fold: int,
        discovery_df: pd.DataFrame,
        train_idx: np.ndarray,
        attention_rows: Sequence[Dict[str, Any]],
        existing_specs: Sequence[ExplicitFeatureSpec],
    ) -> List[ExplicitFeatureSpec]:
        kept_specs: List[ExplicitFeatureSpec] = []
        rejected_low_coverage: List[Dict[str, Any]] = []
        rejected_low_signal: List[Dict[str, Any]] = []
        multivariable_signal_feedback: Dict[str, Any] = {}
        excluded_names: set[str] = set()
        max_attempts = 1 + max(
            0,
            int(self.avf_config.coverage_retry_attempts),
            int(getattr(self.avf_config, "signal_retry_attempts", 0)),
        )

        for attempt in range(1, max_attempts + 1):
            current_specs = self._merge_specs(existing_specs, kept_specs)
            candidates = self._discover_variables_from_attention(
                stage=stage,
                outer_fold=outer_fold,
                discovery_df=discovery_df,
                attention_rows=attention_rows,
                existing_specs=current_specs,
                proposal_attempt=attempt,
                rejected_low_coverage=rejected_low_coverage,
                rejected_low_signal=rejected_low_signal,
                multivariable_signal_feedback=multivariable_signal_feedback,
                excluded_feature_names=sorted(excluded_names),
            )
            current_names = {_normalize_feature_name(spec.name) for spec in current_specs}
            candidates = [
                spec
                for spec in candidates
                if _normalize_feature_name(spec.name) not in current_names
                and _normalize_feature_name(spec.name) not in excluded_names
            ]
            if not candidates:
                break
            candidates = self._harmonize_value_contracts(
                stage=stage,
                outer_fold=outer_fold,
                proposal_attempt=attempt,
                selected_specs=candidates,
            )

            self.dataset = self.extraction_provider.ensure_features(self.dataset, candidates)
            train_df = self.dataset.iloc[train_idx].copy()
            coverage_kept, coverage_dropped = self._partition_specs_by_extraction_coverage(
                train_df,
                candidates,
                manual_specs=[],
            )
            self.coverage_filter_rows.append(
                {
                    "outer_fold": int(outer_fold),
                    "stage": stage,
                    "proposal_attempt": int(attempt),
                    "candidate_features": [spec.name for spec in candidates],
                    "kept_features": [spec.name for spec in coverage_kept],
                    "dropped_features": coverage_dropped,
                }
            )
            self._flush_coverage_filter_rows()

            signal_kept, signal_dropped = self._partition_specs_by_association_signal(
                train_df=train_df,
                stage=stage,
                specs=coverage_kept,
                existing_specs=current_specs,
            )
            kept_specs = self._merge_specs(kept_specs, signal_kept)
            multivariable_signal_feedback = self._multivariable_signal_summary(
                train_df=train_df,
                stage=stage,
                specs=self._merge_specs(existing_specs, kept_specs),
            )
            self.association_filter_rows.append(
                {
                    "outer_fold": int(outer_fold),
                    "stage": stage,
                    "proposal_attempt": int(attempt),
                    "candidate_features": [spec.name for spec in coverage_kept],
                    "kept_features": [spec.name for spec in signal_kept],
                    "dropped_features": signal_dropped,
                    "multivariable_signal": multivariable_signal_feedback,
                }
            )
            self._flush_association_filter_rows()

            dropped = [*coverage_dropped, *signal_dropped]
            signal_inadequate = not bool(multivariable_signal_feedback.get("adequate", True))

            if (not dropped and not signal_inadequate) or attempt >= max_attempts:
                break

            for row in coverage_dropped:
                name = _normalize_feature_name(row.get("name", ""))
                if name:
                    excluded_names.add(name)
                    rejected_low_coverage.append(row)
            for row in signal_dropped:
                name = _normalize_feature_name(row.get("name", ""))
                if name:
                    excluded_names.add(name)
                    rejected_low_signal.append(row)

        return kept_specs

    def _harmonize_value_contracts(
        self,
        stage: str,
        outer_fold: int,
        proposal_attempt: int,
        selected_specs: List[ExplicitFeatureSpec],
    ) -> List[ExplicitFeatureSpec]:
        if not selected_specs:
            return selected_specs
        if not _proposal_agent_supports_value_harmonization(self.proposal_agent):
            return selected_specs

        context = {
            "prompt_version": "multi_model_agentic_value_harmonization_v1",
            "agentic_path": "agentic_attention_variable_forest",
            "stage": stage,
            "outer_fold": int(outer_fold),
            "proposal_attempt": int(proposal_attempt),
            "selected_features": [_spec_to_dict(spec) for spec in selected_specs],
            "missing_value_policy": (
                "Use null for unknown, not reported, not assessed, not tested, "
                "unavailable, and qualitative-only values that are incompatible "
                "with a numeric extraction target."
            ),
        }
        base_row: Dict[str, Any] = {
            "outer_fold": int(outer_fold),
            "stage": stage,
            "proposal_attempt": int(proposal_attempt),
            "selected_features_before": [_spec_to_dict(spec) for spec in selected_specs],
            "selected_features_after": [_spec_to_dict(spec) for spec in selected_specs],
            "applied": [],
        }
        try:
            response = self.proposal_agent.propose(context)
            harmonization_trace = _get_agent_response_trace(self.proposal_agent)
        except Exception as exc:
            logger.warning(
                "Attention-variable value harmonization failed; using unharmonized specs",
                exc_info=True,
            )
            row = {
                **base_row,
                "status": "agent_error_fallback",
                "context": self._stored_agent_context(context),
                "error": str(exc),
            }
            self.value_harmonization_rows.append(row)
            self._flush_value_harmonization_rows()
            return selected_specs

        harmonized, applied = apply_agentic_value_harmonization(
            specs=selected_specs,
            response=response,
        )
        row = {
            **base_row,
            "status": "complete",
            "context": self._stored_agent_context(context),
            "response": response,
            "applied": applied,
            "selected_features_after": [_spec_to_dict(spec) for spec in harmonized],
        }
        if getattr(self.agent_search_config, "save_agent_raw_output", False):
            row["agent_raw_output"] = harmonization_trace
        self.value_harmonization_rows.append(row)
        self._flush_value_harmonization_rows()
        return harmonized

    def _stored_agent_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if getattr(self.agent_search_config, "save_agent_context", False):
            return context
        return _scrub_context(context)

    def _build_agent_context(
        self,
        stage: str,
        outer_fold: int,
        inner_fold: int,
        discovery_df: pd.DataFrame,
        attention_rows: Sequence[Dict[str, Any]],
        existing_specs: Sequence[ExplicitFeatureSpec],
        proposal_attempt: int = 1,
        max_proposals: Optional[int] = None,
        rejected_low_coverage: Optional[Sequence[Dict[str, Any]]] = None,
        rejected_low_signal: Optional[Sequence[Dict[str, Any]]] = None,
        multivariable_signal_feedback: Optional[Dict[str, Any]] = None,
        excluded_feature_names: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        usable_rows = [row for row in attention_rows if _attention_row_has_usable_text(row)]
        if not usable_rows:
            usable_rows = list(attention_rows)
        evidence = sorted(
            usable_rows,
            key=lambda row: abs(float(row.get("attention", 0.0))),
            reverse=True,
        )
        context_rows = [self._attention_evidence_context_row(row) for row in evidence]
        context_rows = [
            row for row in context_rows if row.get("evidence_snippet") or row.get("top_token_spans")
        ]
        instruction = (
            "Infer explicit pre-treatment patient-level variables represented by "
            "repeated high-attention token spans inside high-attention chunks."
        )
        return {
            "prompt_version": "agentic_attention_variable_forest_v1",
            "stage": stage,
            "outer_fold": outer_fold,
            "fold": inner_fold,
            "proposal_attempt": int(proposal_attempt),
            "max_proposals": int(max_proposals or self._candidate_proposal_limit()),
            "instruction": instruction,
            "clinical_question": self.config.clinical_question,
            "estimand": {
                "treatment_column": self.config.treatment_column,
                "outcome_column": self.config.outcome_column,
                "outcome_type": self.config.outcome_type,
            },
            "current_features": [_spec_to_dict(spec) for spec in existing_specs],
            "excluded_feature_names": list(excluded_feature_names or []),
            "rejected_low_coverage_features": list(rejected_low_coverage or []),
            "rejected_low_signal_features": list(rejected_low_signal or []),
            "multivariable_signal_feedback": multivariable_signal_feedback or {},
            "attention_evidence_policy": {
                "source_rows": int(len(attention_rows)),
                "usable_source_rows": int(len(usable_rows)),
                "retained_rows": int(len(context_rows)),
                "row_capacity": None,
                "token_span_capacity_per_row": None,
                "text_character_capacity": None,
                "selection": (
                    "all usable rows in descending absolute chunk-attention order; "
                    "no rows, spans, or text characters are truncated"
                ),
            },
            "signal_source": "attention",
            "attention_evidence": context_rows,
            "fold_label_summary": {
                "n": int(len(discovery_df)),
                "treatment_rate": float(discovery_df[self.config.treatment_column].mean()),
                "outcome_mean": float(discovery_df[self.config.outcome_column].mean()),
            },
            "response_contract": {
                "proposals": [
                    {
                        "action": "add",
                        "name": "snake_case_variable_name",
                        "type": "categorical|continuous",
                        "categories": ["category_a", "category_b"],
                        "description": "exact pre-treatment extraction target",
                        "rationale": "why the attended chunks support this variable",
                    }
                ]
            },
        }

    def _attention_evidence_context_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        spans = _parse_top_token_spans(row.get("top_token_spans_json"))
        compact_spans = _compact_token_spans(spans)
        snippet = _attention_evidence_snippet(
            row.get("chunk_text", ""),
            spans,
            row.get("highlighted_chunk_text"),
        )
        context_row: Dict[str, Any] = {
            "row_id": int(row["row_id"]),
            "attention": _round_context_float(row.get("attention", 0.0)),
        }
        if "chunk_index" in row:
            context_row["chunk_index"] = int(row["chunk_index"])
        if snippet:
            context_row["evidence_snippet"] = snippet
        for key in [
            "e_hat",
            "m_hat",
            "y_residual",
            "t_residual",
            "tau_hat_r_stage",
            "r_pseudo_outcome",
            "r_loss",
        ]:
            if key in row:
                context_row[key] = _round_context_float(row[key])
        if compact_spans:
            context_row["top_token_spans"] = compact_spans
            summary = row.get("attended_token_summary")
            if isinstance(summary, str) and summary:
                context_row["attended_token_summary"] = _normalize_context_text(summary)
        return context_row

    def _candidate_proposal_limit(self) -> int:
        configured = int(getattr(self.avf_config, "candidate_proposals_per_fold", 3))
        agent_limit = int(getattr(self.agent_search_config, "max_additions_per_iter", configured))
        return max(1, min(configured, max(1, agent_limit)))

    def _fit_final_forest(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        selected_specs: List[ExplicitFeatureSpec],
        fold_id: int,
    ) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
        train_T = train_df[self.config.treatment_column].to_numpy()
        train_Y = train_df[self.config.outcome_column].to_numpy()
        test_T = test_df[self.config.treatment_column].to_numpy()
        test_Y = test_df[self.config.outcome_column].to_numpy()

        X_train, W_train, x_names, w_names, means, stds = _build_features(
            train_df,
            selected_specs,
        )
        X_test, W_test, _, _, _, _ = _build_features(test_df, selected_specs, means, stds)
        actual_x_dim = 0 if X_train is None else X_train.shape[1]
        if X_train is None:
            X_train = np.zeros((len(train_df), 1), dtype=np.float32)
            X_test = np.zeros((len(test_df), 1), dtype=np.float32)
            x_names = ["intercept_effect"]

        fitted = fit_predict_explicit_forest(
            X_train=X_train,
            W_train=W_train,
            T_train=train_T,
            Y_train=train_Y,
            X_test=X_test,
            W_test=W_test,
            T_test=test_T,
            Y_test=test_Y,
            cf_config=self.cf_config,
            outcome_type=self.config.outcome_type,
            random_state=42 + fold_id,
        )
        tau = fitted["tau"]

        nuisance_train = _hstack_present(X_train, W_train)
        nuisance_test = _hstack_present(X_test, W_test)
        if nuisance_train is None:
            nuisance_train = np.zeros((len(train_df), 1), dtype=np.float32)
            nuisance_test = np.zeros((len(test_df), 1), dtype=np.float32)

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

        y0 = outcome_pred - propensity * tau
        y1 = outcome_pred + (1.0 - propensity) * tau
        if self.config.outcome_type == "binary":
            y0 = np.clip(y0, 0.0, 1.0)
            y1 = np.clip(y1, 0.0, 1.0)

        predictions = add_forest_prediction_columns(test_df.copy(), fitted)
        predictions["pred_y0_prob"] = y0
        predictions["pred_y1_prob"] = y1
        predictions["pred_propensity_prob"] = propensity
        predictions["pred_outcome_prob"] = outcome_pred
        predictions["cv_fold"] = fold_id

        metrics = {
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "n_selected_features": int(len(selected_specs)),
            "n_x_features": int(actual_x_dim),
            "n_w_features": int(0 if W_train is None else W_train.shape[1]),
            **fitted["ate_metrics"],
            "treatment_auroc": _safe_roc_auc(test_T, propensity),
            "x_feature_names": x_names,
            "w_feature_names": w_names,
        }
        return predictions, metrics, fitted["fit_audit"]

    @staticmethod
    def _merge_specs(*spec_groups: Sequence[ExplicitFeatureSpec]) -> List[ExplicitFeatureSpec]:
        merged: Dict[str, ExplicitFeatureSpec] = {}
        for group in spec_groups:
            for spec in group:
                name = _normalize_feature_name(spec.name)
                if name in merged:
                    roles = list(dict.fromkeys([*merged[name].roles, *spec.roles]))
                    merged[name] = ExplicitFeatureSpec(
                        name=merged[name].name,
                        type=merged[name].type,
                        categories=merged[name].categories,
                        description=merged[name].description or spec.description,
                        value_aliases=getattr(merged[name], "value_aliases", None),
                        roles=roles,
                    )
                else:
                    merged[name] = spec
        return list(merged.values())

    def _fold_n_jobs(self, folds: int) -> int:
        return self._parallel_n_jobs(
            self.avf_config.fold_parallelism,
            folds,
            auto_workers=(
                len(self.devices)
                if self.device.type != "cpu" and len(self.devices) > 1
                else self.num_workers
            ),
            cuda_auto_serial=len(self.devices) <= 1,
        )

    def _outer_n_jobs(self, folds: int) -> int:
        return self._parallel_n_jobs(
            self.avf_config.outer_parallelism,
            folds,
            auto_workers=(
                min(max(1, int(self.num_workers)), len(self.devices))
                if self.device.type != "cpu" and len(self.devices) > 1
                else self.num_workers
            ),
            cuda_auto_serial=False,
        )

    def _parallel_n_jobs(
        self,
        setting: Any,
        tasks: int,
        *,
        auto_workers: int,
        cuda_auto_serial: bool,
    ) -> int:
        if tasks <= 0:
            return 1
        setting_text = str(setting).strip().lower()
        if setting_text == "auto":
            if cuda_auto_serial and self.device.type != "cpu":
                return 1
            return max(1, min(int(auto_workers), int(tasks)))
        return max(1, min(int(setting_text), int(tasks)))

    def _inner_workers_for_outer_job(self, outer_n_jobs: int) -> int:
        if str(self.avf_config.fold_parallelism).strip().lower() != "auto":
            return self.num_workers
        return max(1, int(self.num_workers) // max(1, int(outer_n_jobs)))

    def _agent_candidate_n_jobs(self, folds: int) -> int:
        n_jobs = self._parallel_n_jobs(
            getattr(self.avf_config, "candidate_proposal_parallelism", "1"),
            folds,
            auto_workers=self.num_workers,
            cuda_auto_serial=False,
        )
        if n_jobs > 1 and self._has_custom_proposal_agent:
            logger.warning(
                "Agent candidate proposal parallelism disabled because a custom "
                "proposal_agent object was supplied and may not be thread-safe."
            )
            return 1
        return n_jobs

    def _device_for_inner_fold(self, fold: int) -> torch.device:
        if not self.devices:
            return self.device
        return self.devices[(int(fold) - 1) % len(self.devices)]

    def _device_context_for_inner_fold(self, fold: int):
        return self._using_device(self._device_for_inner_fold(fold))

    def _outer_device_groups(self, outer_n_jobs: int) -> List[List[torch.device]]:
        if not self.devices:
            return [[self.device]]
        if outer_n_jobs <= 1:
            return [list(self.devices)]
        group_count = min(max(1, int(outer_n_jobs)), len(self.devices))
        groups: List[List[torch.device]] = [[] for _ in range(group_count)]
        for index, device in enumerate(self.devices):
            groups[index % group_count].append(device)
        return [group for group in groups if group]

    def _outer_device_group_queue(self, outer_n_jobs: int):
        device_group_queue = queue.Queue()
        for devices in self._outer_device_groups(outer_n_jobs):
            device_group_queue.put(devices)
        return device_group_queue

    def _data_loader_workers(self, total_folds: Optional[int] = None) -> int:
        """Apply standalone fold/thread guards to the shared loader worker budget."""
        env_workers = os.environ.get("OCI_AVF_DATALOADER_WORKERS")
        if env_workers is not None:
            return max(0, int(env_workers))
        if _running_inside_loky_worker():
            return 0
        # Cross-fit and outer-fold jobs run in thread pools.  Starting a
        # multiprocessing DataLoader from one of those threads creates nested
        # process parallelism, can oversubscribe the configured worker budget,
        # and is not supported by every multiprocessing start method.
        if threading.current_thread() is not threading.main_thread():
            return 0
        if total_folds is not None and self._fold_n_jobs(total_folds) > 1:
            return 0
        return super()._data_loader_workers(total_folds=total_folds)

    def _crossfit_checkpoint_fingerprint(
        self,
        stage: str,
        folds: int,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> str:
        arch = self.config.architecture
        train = self.config.training
        arch_keys = [
            "feature_extractor_type",
            "htr_sentence_model",
            "htr_freeze_sentence_encoder",
            "htr_chunk_size_words",
            "htr_chunk_overlap_words",
            "htr_max_chunks",
            "htr_max_chunk_length",
            "htr_num_layers",
            "htr_num_heads",
            "htr_transformer_dim",
            "htr_dropout",
            "htr_projection_dim",
            "htr_hash_embedding_dim",
            "htr_sentence_encoder_batch_size",
            "htr_sentence_encoder_backend",
            "htr_sentence_pooling",
            "htr_normalize_sentence_embeddings",
            "htr_trainable_sentence_encoder_layers",
            "htr_prediction_head_hidden_dim",
            *HTR_NUMERIC_SUMMARY_KEYS,
        ]
        train_keys = [
            "epochs",
            "batch_size",
            "effect_batch_size",
            "learning_rate",
            "weight_decay",
            "gradient_clip_norm",
            "alpha_propensity",
            "lr_schedule",
        ]
        payload = {
            "stage": stage,
            "folds": int(folds),
            "outcome_type": self.config.outcome_type,
            "text_column": self.config.text_column,
            "outcome_column": self.config.outcome_column,
            "treatment_column": self.config.treatment_column,
            "attention_top_k_chunks": self.avf_config.attention_top_k_chunks,
            "e_clip": self.avf_config.e_clip,
            "nuisance_epochs": self.avf_config.nuisance_epochs,
            "nuisance_weight_decay": self.avf_config.nuisance_weight_decay,
            "nuisance_label_smoothing": self.avf_config.nuisance_label_smoothing,
            "nuisance_calibration": self.avf_config.nuisance_calibration,
            "effect_epochs": self.avf_config.effect_epochs,
            "architecture": omit_inactive_htr_numeric_settings(
                {key: getattr(arch, key, None) for key in arch_keys}
            ),
            "training": {key: getattr(train, key, None) for key in train_keys},
        }
        if extra_payload:
            payload["extra"] = extra_payload
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _fold_checkpoint_paths(
        self,
        stage: str,
        outer_fold: int,
        fold: int,
    ) -> Dict[str, Path]:
        stage_dir = self.artifact_dir / "crossfit_fold_checkpoints" / stage
        stem = f"outer_{int(outer_fold):03d}_fold_{int(fold):03d}"
        return {
            "predictions": stage_dir / f"{stem}_predictions.parquet",
            "attention": stage_dir / f"{stem}_attention.parquet",
            "done": stage_dir / f"{stem}.done.json",
        }

    def _load_fold_checkpoint(
        self,
        stage: str,
        df: pd.DataFrame,
        outer_fold: int,
        fold: int,
        heldout_pos: np.ndarray,
        fingerprint: str,
    ) -> Optional[Tuple[pd.DataFrame, List[Dict[str, Any]]]]:
        paths = self._fold_checkpoint_paths(stage, outer_fold, fold)
        if not (
            paths["done"].exists() and paths["predictions"].exists() and paths["attention"].exists()
        ):
            return None
        try:
            with open(paths["done"]) as f:
                marker = json.load(f)
            if marker.get("fingerprint") != fingerprint:
                logger.info(
                    "Ignoring stale %s checkpoint for outer fold %s fold %s: "
                    "fingerprint mismatch",
                    stage,
                    outer_fold,
                    fold,
                )
                return None

            pred_df = pd.read_parquet(paths["predictions"])
            attention_df = pd.read_parquet(paths["attention"])
            expected_ids = df.iloc[heldout_pos]["_oci_row_id"].to_numpy()
            if "_oci_row_id" not in pred_df.columns or pred_df["_oci_row_id"].duplicated().any():
                logger.warning(
                    "Ignoring invalid %s checkpoint for outer fold %s fold %s: "
                    "missing or duplicate _oci_row_id",
                    stage,
                    outer_fold,
                    fold,
                )
                return None
            pred_by_id = pred_df.set_index("_oci_row_id", drop=False)
            if not set(expected_ids).issubset(set(pred_by_id.index)):
                logger.info(
                    "Ignoring stale %s checkpoint for outer fold %s fold %s: "
                    "heldout row IDs changed",
                    stage,
                    outer_fold,
                    fold,
                )
                return None
            pred_df = pred_by_id.loc[expected_ids].reset_index(drop=True)
            if len(pred_df) != len(expected_ids):
                logger.info(
                    "Ignoring stale %s checkpoint for outer fold %s fold %s: "
                    "heldout row count changed",
                    stage,
                    outer_fold,
                    fold,
                )
                return None
            attention_rows = attention_df.to_dict("records")
            logger.info(
                "Outer fold %s %s fold %s: loaded cached checkpoint "
                "predictions=%s attention_rows=%s",
                outer_fold,
                stage,
                fold,
                len(pred_df),
                len(attention_rows),
            )
            return pred_df, attention_rows
        except Exception as exc:
            logger.warning(
                "Ignoring unreadable %s checkpoint for outer fold %s fold %s: %s",
                stage,
                outer_fold,
                fold,
                exc,
            )
            return None

    def _save_fold_checkpoint(
        self,
        stage: str,
        outer_fold: int,
        fold: int,
        predictions: pd.DataFrame,
        attention_rows: Sequence[Dict[str, Any]],
        fingerprint: str,
    ) -> None:
        paths = self._fold_checkpoint_paths(stage, outer_fold, fold)
        paths["predictions"].parent.mkdir(parents=True, exist_ok=True)
        attention_df = pd.DataFrame(attention_rows)
        if attention_df.empty:
            attention_df = pd.DataFrame(columns=["row_id", "fold", "stage", "outer_fold"])
        _write_parquet_atomic(predictions, paths["predictions"])
        _write_parquet_atomic(attention_df, paths["attention"])
        _write_json_atomic(
            {
                "stage": stage,
                "outer_fold": int(outer_fold),
                "fold": int(fold),
                "n_predictions": int(len(predictions)),
                "n_attention_rows": int(len(attention_df)),
                "fingerprint": fingerprint,
            },
            paths["done"],
        )
        logger.info(
            "Outer fold %s %s fold %s: saved checkpoint predictions=%s " "attention_rows=%s",
            outer_fold,
            stage,
            fold,
            len(predictions),
            len(attention_df),
        )

    def _agent_candidate_checkpoint_path(
        self,
        stage: str,
        outer_fold: int,
        fold: int,
        proposal_attempt: int = 1,
    ) -> Path:
        stage_dir = self.artifact_dir / "agent_candidate_checkpoints" / stage
        stem = f"outer_{int(outer_fold):03d}_fold_{int(fold):03d}"
        if int(proposal_attempt) > 1:
            stem = f"{stem}_attempt_{int(proposal_attempt):03d}"
        return stage_dir / f"{stem}.json"

    def _save_agent_candidate_checkpoint(
        self,
        row: Dict[str, Any],
        stage: str,
        outer_fold: int,
        fold: int,
    ) -> None:
        path = self._agent_candidate_checkpoint_path(
            stage,
            outer_fold,
            fold,
            proposal_attempt=int(row.get("proposal_attempt", 1)),
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(row, path)
        logger.info(
            "Outer fold %s %s fold %s: saved agent checkpoint status=%s path=%s",
            outer_fold,
            stage,
            fold,
            row.get("status", "unknown"),
            path,
        )

    def _agent_consensus_checkpoint_path(
        self,
        stage: str,
        outer_fold: int,
        proposal_attempt: int = 1,
    ) -> Path:
        stage_dir = self.artifact_dir / "agent_candidate_checkpoints" / stage
        stem = f"outer_{int(outer_fold):03d}_consensus_attempt_" f"{int(proposal_attempt):03d}"
        return stage_dir / f"{stem}.json"

    def _save_agent_consensus_checkpoint(
        self,
        row: Dict[str, Any],
        stage: str,
        outer_fold: int,
        proposal_attempt: int,
    ) -> None:
        path = self._agent_consensus_checkpoint_path(
            stage=stage,
            outer_fold=outer_fold,
            proposal_attempt=proposal_attempt,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(row, path)
        logger.info(
            "Outer fold %s %s consensus attempt %s: saved agent checkpoint " "status=%s path=%s",
            outer_fold,
            stage,
            proposal_attempt,
            row.get("status", "unknown"),
            path,
        )

    def _agent_consensus_recovery_checkpoint_path(
        self,
        stage: str,
        outer_fold: int,
        proposal_attempt: int = 1,
    ) -> Path:
        stage_dir = self.artifact_dir / "agent_candidate_checkpoints" / stage
        stem = (
            f"outer_{int(outer_fold):03d}_consensus_recovery_attempt_"
            f"{int(proposal_attempt):03d}"
        )
        return stage_dir / f"{stem}.json"

    def _save_agent_consensus_recovery_checkpoint(
        self,
        row: Dict[str, Any],
        stage: str,
        outer_fold: int,
        proposal_attempt: int,
    ) -> None:
        path = self._agent_consensus_recovery_checkpoint_path(
            stage=stage,
            outer_fold=outer_fold,
            proposal_attempt=proposal_attempt,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(row, path)
        logger.info(
            "Outer fold %s %s consensus recovery attempt %s: saved agent "
            "checkpoint status=%s path=%s",
            outer_fold,
            stage,
            proposal_attempt,
            row.get("status", "unknown"),
            path,
        )

    def _build_consensus_disambiguation_context(
        self,
        stage: str,
        outer_fold: int,
        proposal_attempt: int,
        proposal_artifacts_by_fold: Dict[int, List[Dict[str, Any]]],
        threshold: int,
        excluded_feature_names: Sequence[str],
    ) -> Dict[str, Any]:
        proposed_by_fold: List[Dict[str, Any]] = []
        for fold in sorted(proposal_artifacts_by_fold):
            proposals = []
            for proposal in proposal_artifacts_by_fold.get(fold, []):
                name = _normalize_feature_name(proposal.get("name"))
                if not name:
                    continue
                proposals.append(
                    {
                        "fold": int(fold),
                        "name": name,
                        "type": proposal.get("type"),
                        "categories": proposal.get("categories"),
                        "description": proposal.get("description"),
                        "rationale": proposal.get("rationale"),
                        "roles": list(proposal.get("roles") or [stage]),
                    }
                )
            proposed_by_fold.append({"fold": int(fold), "proposals": proposals})

        return {
            "prompt_version": "agentic_attention_consensus_disambiguation_v1",
            "stage": stage,
            "required_role": stage,
            "outer_fold": int(outer_fold),
            "proposal_attempt": int(proposal_attempt),
            "max_proposals": self._candidate_proposal_limit(),
            "fold_count": int(len(proposal_artifacts_by_fold)),
            "consensus_threshold": int(threshold),
            "excluded_feature_names": list(excluded_feature_names),
            "proposed_variables_by_fold": proposed_by_fold,
        }

    def _resolve_consensus_disambiguation(
        self,
        stage: str,
        outer_fold: int,
        proposal_attempt: int,
        proposals_by_fold: Dict[int, List[ExplicitFeatureSpec]],
        proposal_artifacts_by_fold: Dict[int, List[Dict[str, Any]]],
        excluded_feature_names: Sequence[str],
    ) -> Dict[str, Any]:
        fold_count = len(proposals_by_fold)
        threshold = _consensus_threshold(
            fold_count=fold_count,
            min_fold_fraction=self.avf_config.consensus_min_fold_fraction,
            min_folds=getattr(self.avf_config, "consensus_min_folds", None),
        )
        raw_proposals = _proposals_by_fold_artifact(proposal_artifacts_by_fold)
        base_row: Dict[str, Any] = {
            "outer_fold": int(outer_fold),
            "stage": stage,
            "proposal_attempt": int(proposal_attempt),
            "threshold": int(threshold),
            "fold_count": int(fold_count),
            "raw_proposals": raw_proposals,
            "validated_groups": [],
            "selected_groups": [],
            "validation_errors": [],
        }
        folds_with_proposals = [
            fold
            for fold, specs in proposals_by_fold.items()
            if any(stage in spec.roles for spec in specs)
        ]
        if threshold < 2 or len(folds_with_proposals) < 2:
            base_row["status"] = "skipped_insufficient_fold_support"
            return base_row

        context = self._build_consensus_disambiguation_context(
            stage=stage,
            outer_fold=outer_fold,
            proposal_attempt=proposal_attempt,
            proposal_artifacts_by_fold=proposal_artifacts_by_fold,
            threshold=threshold,
            excluded_feature_names=excluded_feature_names,
        )
        started_row = {
            **base_row,
            "status": "started",
            "context": self._stored_agent_context(context),
        }
        self._save_agent_consensus_checkpoint(
            started_row,
            stage=stage,
            outer_fold=outer_fold,
            proposal_attempt=proposal_attempt,
        )

        try:
            raw_response = self.proposal_agent.propose(context)
        except Exception as exc:
            error_row = {
                **base_row,
                "status": "agent_error_fallback",
                "context": self._stored_agent_context(context),
                "error": str(exc),
            }
            if getattr(self.agent_search_config, "save_agent_raw_output", False):
                error_row["agent_raw_output"] = _get_agent_response_trace(self.proposal_agent)
            self._save_agent_consensus_checkpoint(
                error_row,
                stage=stage,
                outer_fold=outer_fold,
                proposal_attempt=proposal_attempt,
            )
            return error_row

        validated_groups, validation_errors = _validate_consensus_disambiguation_response(
            raw_response,
            proposals_by_fold=proposals_by_fold,
            required_role=stage,
        )
        status = "complete"
        if not isinstance(raw_response, dict):
            status = "invalid_response_fallback"
        elif validation_errors and not validated_groups:
            status = "invalid_groups_fallback"
        row = {
            **base_row,
            "status": status,
            "context": self._stored_agent_context(context),
            "raw_response": raw_response,
            "validated_groups": validated_groups,
            "validation_errors": validation_errors,
        }
        if getattr(self.agent_search_config, "save_agent_raw_output", False):
            row["agent_raw_output"] = _get_agent_response_trace(self.proposal_agent)
        self._save_agent_consensus_checkpoint(
            row,
            stage=stage,
            outer_fold=outer_fold,
            proposal_attempt=proposal_attempt,
        )
        return row

    def _flush_agent_candidate_rows(self) -> None:
        _write_jsonl(
            self.artifact_dir / "confounder_candidates_by_fold.jsonl",
            self.confounder_candidate_rows,
        )
        _write_jsonl(
            self.artifact_dir / "effect_modifier_candidates_by_fold.jsonl",
            self.modifier_candidate_rows,
        )

    def _flush_consensus_disambiguation_rows(self) -> None:
        _write_jsonl(
            self.artifact_dir / "consensus_disambiguation_by_attempt.jsonl",
            self.consensus_disambiguation_rows,
        )

    def _flush_consensus_recovery_rows(self) -> None:
        _write_jsonl(
            self.artifact_dir / "consensus_recovery_by_attempt.jsonl",
            self.consensus_recovery_rows,
        )

    def _flush_value_harmonization_rows(self) -> None:
        _write_jsonl(
            self.artifact_dir / "value_harmonization_by_attempt.jsonl",
            self.value_harmonization_rows,
        )

    def _flush_coverage_filter_rows(self) -> None:
        _write_jsonl(
            self.artifact_dir / "coverage_filter_by_attempt.jsonl",
            self.coverage_filter_rows,
        )

    def _flush_association_filter_rows(self) -> None:
        _write_jsonl(
            self.artifact_dir / "association_filter_by_attempt.jsonl",
            self.association_filter_rows,
        )

    def _load_nuisance_fold_checkpoint(
        self,
        df: pd.DataFrame,
        outer_fold: int,
        fold: int,
        heldout_pos: np.ndarray,
        fingerprint: str,
    ) -> Optional[Dict[str, Any]]:
        loaded = self._load_fold_checkpoint(
            "nuisance",
            df,
            outer_fold,
            fold,
            heldout_pos,
            fingerprint,
        )
        if loaded is None:
            return None
        pred_df, attention_rows = loaded
        return {
            "fold": fold,
            "heldout_pos": np.asarray(heldout_pos),
            "e_hat_raw": pred_df["e_hat_raw"].to_numpy(dtype=float),
            "m_hat_raw": pred_df["m_hat_raw"].to_numpy(dtype=float),
            "attention": attention_rows,
        }

    def _save_nuisance_fold_checkpoint(
        self,
        df: pd.DataFrame,
        result: Dict[str, Any],
        outer_fold: int,
        fingerprint: str,
    ) -> None:
        heldout_pos = np.asarray(result["heldout_pos"], dtype=int)
        predictions = pd.DataFrame(
            {
                "heldout_pos": heldout_pos,
                "_oci_row_id": df.iloc[heldout_pos]["_oci_row_id"].to_numpy(),
                "outer_fold": int(outer_fold),
                "nuisance_fold": int(result["fold"]),
                # Raw predictions only: calibration is cross-fitted across
                # folds after all folds finish and is recomputed on resume.
                "e_hat_raw": np.asarray(result["e_hat_raw"], dtype=float),
                "m_hat_raw": np.asarray(result["m_hat_raw"], dtype=float),
            }
        )
        self._save_fold_checkpoint(
            "nuisance",
            outer_fold,
            int(result["fold"]),
            predictions,
            result["attention"],
            fingerprint,
        )

    def _load_effect_fold_checkpoint(
        self,
        df: pd.DataFrame,
        outer_fold: int,
        fold: int,
        heldout_pos: np.ndarray,
        fingerprint: str,
    ) -> Optional[Dict[str, Any]]:
        loaded = self._load_fold_checkpoint(
            "r_stage",
            df,
            outer_fold,
            fold,
            heldout_pos,
            fingerprint,
        )
        if loaded is None:
            return None
        pred_df, attention_rows = loaded
        default_effect_loss = (
            pred_df["effect_loss"].to_numpy(dtype=float)
            if "effect_loss" in pred_df.columns
            else pred_df["r_loss"].to_numpy(dtype=float)
        )
        default_effect_loss_at_zero = (
            pred_df["effect_loss_at_zero_tau"].to_numpy(dtype=float)
            if "effect_loss_at_zero_tau" in pred_df.columns
            else np.full(len(pred_df), np.nan, dtype=float)
        )
        return {
            "fold": fold,
            "heldout_pos": np.asarray(heldout_pos),
            "tau_hat": pred_df["tau_hat_r_stage"].to_numpy(dtype=float),
            "tau_logit_modifier": (
                pred_df["tau_logit_modifier"].to_numpy(dtype=float)
                if "tau_logit_modifier" in pred_df.columns
                else np.full(len(pred_df), np.nan, dtype=float)
            ),
            "r_loss": pred_df["r_loss"].to_numpy(dtype=float),
            "effect_loss": default_effect_loss,
            "effect_loss_at_zero_tau": default_effect_loss_at_zero,
            "effect_objective": (
                str(pred_df["effect_objective"].iloc[0])
                if "effect_objective" in pred_df.columns and len(pred_df) > 0
                else effect_objective_name(self.avf_config)
            ),
            "attention": attention_rows,
        }

    def _save_effect_fold_checkpoint(
        self,
        df: pd.DataFrame,
        result: Dict[str, Any],
        outer_fold: int,
        fingerprint: str,
    ) -> None:
        heldout_pos = np.asarray(result["heldout_pos"], dtype=int)
        predictions = pd.DataFrame(
            {
                "heldout_pos": heldout_pos,
                "_oci_row_id": df.iloc[heldout_pos]["_oci_row_id"].to_numpy(),
                "outer_fold": int(outer_fold),
                "effect_fold": int(result["fold"]),
                "tau_hat_r_stage": np.asarray(result["tau_hat"], dtype=float),
                "tau_logit_modifier": np.asarray(
                    result.get(
                        "tau_logit_modifier",
                        np.full(len(heldout_pos), np.nan),
                    ),
                    dtype=float,
                ),
                "r_loss": np.asarray(result["r_loss"], dtype=float),
                "effect_loss": np.asarray(
                    result.get("effect_loss", result["r_loss"]),
                    dtype=float,
                ),
                "effect_loss_at_zero_tau": np.asarray(
                    result.get(
                        "effect_loss_at_zero_tau",
                        np.full(len(heldout_pos), np.nan),
                    ),
                    dtype=float,
                ),
                "effect_objective": np.asarray(
                    [
                        str(
                            result.get(
                                "effect_objective",
                                effect_objective_name(self.avf_config),
                            )
                        )
                    ]
                    * len(heldout_pos),
                    dtype=object,
                ),
                "r_stage_train_eligible": np.asarray(
                    result.get(
                        "r_stage_train_eligible",
                        np.ones(len(heldout_pos), dtype=bool),
                    ),
                    dtype=bool,
                ),
            }
        )
        self._save_fold_checkpoint(
            "r_stage",
            outer_fold,
            int(result["fold"]),
            predictions,
            result["attention"],
            fingerprint,
        )

    def _filter_specs_by_extraction_coverage(
        self,
        df: pd.DataFrame,
        specs: Sequence[ExplicitFeatureSpec],
        manual_specs: Sequence[ExplicitFeatureSpec],
    ) -> List[ExplicitFeatureSpec]:
        kept, _ = self._partition_specs_by_extraction_coverage(
            df,
            specs,
            manual_specs,
        )
        return kept

    def _partition_specs_by_extraction_coverage(
        self,
        df: pd.DataFrame,
        specs: Sequence[ExplicitFeatureSpec],
        manual_specs: Sequence[ExplicitFeatureSpec],
    ) -> Tuple[List[ExplicitFeatureSpec], List[Dict[str, Any]]]:
        manual_names = {_normalize_feature_name(spec.name) for spec in manual_specs}
        kept: List[ExplicitFeatureSpec] = []
        dropped: List[Dict[str, Any]] = []
        for spec in specs:
            name = _normalize_feature_name(spec.name)
            coverage = _feature_coverage(df, name)
            if coverage < self.avf_config.min_extraction_coverage and not (
                self.avf_config.manual_features_locked and name in manual_names
            ):
                logger.info(
                    "Dropping discovered feature %s for low extraction coverage %.3f < %.3f",
                    name,
                    coverage,
                    self.avf_config.min_extraction_coverage,
                )
                dropped.append(
                    {
                        "name": name,
                        "type": spec.type,
                        "roles": list(spec.roles),
                        "description": spec.description,
                        "coverage": float(coverage),
                        "min_extraction_coverage": float(self.avf_config.min_extraction_coverage),
                    }
                )
                continue
            kept.append(spec)
        return kept, dropped

    def _partition_specs_by_association_signal(
        self,
        train_df: pd.DataFrame,
        stage: str,
        specs: Sequence[ExplicitFeatureSpec],
        existing_specs: Sequence[ExplicitFeatureSpec],
    ) -> Tuple[List[ExplicitFeatureSpec], List[Dict[str, Any]]]:
        kept: List[ExplicitFeatureSpec] = []
        dropped: List[Dict[str, Any]] = []
        alpha = float(getattr(self.avf_config, "association_alpha", 0.05))
        for spec in specs:
            diagnostic = _feature_association_diagnostic(
                df=train_df,
                spec=spec,
                config=self.config,
                existing_specs=existing_specs,
                alpha=alpha,
                min_n=int(getattr(self.avf_config, "association_min_n", 20)),
                min_non_missing=int(getattr(self.avf_config, "association_min_non_missing", 10)),
            )
            if diagnostic.get("status") == "skipped_insufficient_sample":
                kept.append(spec)
                continue

            if stage == "confounder":
                keep = bool(
                    diagnostic.get("treatment_associated") and diagnostic.get("outcome_associated")
                )
                rejection_reason = "no_joint_treatment_outcome_association"
            else:
                keep = bool(
                    diagnostic.get("outcome_associated") or diagnostic.get("interaction_associated")
                )
                rejection_reason = "no_outcome_or_interaction_association"

            if keep:
                kept.append(spec)
                continue

            dropped.append(
                {
                    "name": spec.name,
                    "type": spec.type,
                    "roles": list(spec.roles),
                    "description": spec.description,
                    "rejection_reason": rejection_reason,
                    "diagnostic": diagnostic,
                }
            )
        return kept, dropped

    def _multivariable_signal_summary(
        self,
        train_df: pd.DataFrame,
        stage: str,
        specs: Sequence[ExplicitFeatureSpec],
    ) -> Dict[str, Any]:
        specs = list(specs)
        if not specs:
            return {
                "status": "no_features",
                "adequate": False,
                "reason": "no_features_survived_association_screen",
            }
        min_n = int(getattr(self.avf_config, "association_min_n", 20))
        if len(train_df) < min_n:
            return {
                "status": "skipped_insufficient_sample",
                "adequate": True,
                "n": int(len(train_df)),
                "min_n": min_n,
            }

        matrix, feature_names = _signal_feature_matrix(train_df, specs)
        if matrix is None or matrix.shape[1] == 0 or not _has_any_variation(matrix):
            return {
                "status": "no_varying_features",
                "adequate": False,
                "feature_names": feature_names,
            }

        folds = int(getattr(self.avf_config, "signal_cv_folds", 3))
        treatment_score = _cross_validated_boosted_signal_score(
            matrix,
            train_df[self.config.treatment_column].to_numpy(),
            target_kind="binary",
            folds=folds,
            random_state=71,
        )
        outcome_score = _cross_validated_boosted_signal_score(
            matrix,
            train_df[self.config.outcome_column].to_numpy(),
            target_kind=self.config.outcome_type,
            folds=folds,
            random_state=173,
        )
        min_treatment = float(getattr(self.avf_config, "min_signal_treatment_auroc", 0.55))
        min_outcome = float(getattr(self.avf_config, "min_signal_outcome_auroc", 0.55))
        treatment_ok = _score_meets_signal_threshold(treatment_score, min_treatment)
        outcome_ok = _score_meets_signal_threshold(outcome_score, min_outcome)
        adequate = bool((treatment_ok and outcome_ok) if stage == "confounder" else outcome_ok)
        return {
            "status": "ok",
            "adequate": adequate,
            "stage": stage,
            "required": (
                "treatment_and_outcome_auroc" if stage == "confounder" else "outcome_auroc"
            ),
            "min_treatment_auroc": min_treatment,
            "min_outcome_auroc": min_outcome,
            "treatment_ok": bool(treatment_ok),
            "outcome_ok": bool(outcome_ok),
            "treatment_model": treatment_score,
            "outcome_model": outcome_score,
            "feature_names": feature_names,
            "features": [spec.name for spec in specs],
        }

    def _save_predictions(self, results_df: pd.DataFrame) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_parquet(self.output_path, index=False)
        logger.info("Predictions saved to: %s", self.output_path)

    def _save_artifacts(self, results_df: pd.DataFrame) -> None:
        if self.nuisance_rows:
            pd.concat(self.nuisance_rows).to_parquet(
                self.artifact_dir / "nuisance_oof_predictions.parquet",
                index=False,
            )
        if self.r_stage_rows:
            pd.concat(self.r_stage_rows).to_parquet(
                self.artifact_dir / "r_stage_oof_predictions.parquet",
                index=False,
            )
        pd.DataFrame(self.nuisance_attention_rows).to_parquet(
            self.artifact_dir / "nuisance_attention_evidence.parquet",
            index=False,
        )
        pd.DataFrame(self.effect_attention_rows).to_parquet(
            self.artifact_dir / "r_stage_attention_evidence.parquet",
            index=False,
        )
        _write_jsonl(
            self.artifact_dir / "confounder_candidates_by_fold.jsonl",
            self.confounder_candidate_rows,
        )
        _write_jsonl(
            self.artifact_dir / "effect_modifier_candidates_by_fold.jsonl",
            self.modifier_candidate_rows,
        )
        _write_jsonl(
            self.artifact_dir / "consensus_disambiguation_by_attempt.jsonl",
            self.consensus_disambiguation_rows,
        )
        _write_jsonl(
            self.artifact_dir / "consensus_recovery_by_attempt.jsonl",
            self.consensus_recovery_rows,
        )
        _write_jsonl(
            self.artifact_dir / "value_harmonization_by_attempt.jsonl",
            self.value_harmonization_rows,
        )
        with open(self.artifact_dir / "consensus.json", "w") as f:
            json.dump(self.consensus_rows, f, indent=2)
        _write_jsonl(
            self.artifact_dir / "coverage_filter_by_attempt.jsonl",
            self.coverage_filter_rows,
        )
        _write_jsonl(
            self.artifact_dir / "association_filter_by_attempt.jsonl",
            self.association_filter_rows,
        )
        metrics_for_csv = [
            {key: value for key, value in row.items() if not isinstance(value, list)}
            for row in self.metric_rows
        ]
        pd.DataFrame(metrics_for_csv).to_csv(self.artifact_dir / "metrics.csv", index=False)
        manifest = {
            "model_type": "agentic_attention_variable_forest",
            "config": asdict(self.avf_config),
            "n_rows": int(len(results_df)),
            "output_path": str(self.output_path),
        }
        with open(self.artifact_dir / "run_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)


def _consensus_threshold(
    fold_count: int,
    min_fold_fraction: float,
    min_folds: Optional[int] = None,
) -> int:
    """Return the fold count required for consensus."""
    fold_count = max(1, int(fold_count))
    if min_folds is not None:
        return min(fold_count, max(1, int(min_folds)))
    return max(1, int(np.ceil(float(min_fold_fraction) * fold_count)))


def consensus_feature_specs(
    proposals_by_fold: Dict[int, Sequence[ExplicitFeatureSpec]],
    min_fold_fraction: float,
    required_role: str,
    min_folds: Optional[int] = None,
    concept_groups: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[ExplicitFeatureSpec]:
    """Select specs whose normalized concept recurs across enough folds."""
    fold_count = max(1, len(proposals_by_fold))
    threshold = _consensus_threshold(fold_count, min_fold_fraction, min_folds)
    if concept_groups is not None:
        selected_from_groups: List[ExplicitFeatureSpec] = []
        for group in sorted(
            concept_groups,
            key=lambda row: str(row.get("canonical_name", "")),
        ):
            folds = {int(fold) for fold in group.get("member_folds", []) if str(fold).strip()}
            if len(folds) < threshold:
                continue
            roles = [role for role in group.get("roles", []) if role in VALID_ROLES]
            if required_role not in roles:
                roles.append(required_role)
            name = _normalize_feature_name(group.get("canonical_name"))
            typ = str(group.get("type") or "categorical").lower()
            if not name or typ not in VALID_TYPES:
                continue
            categories = group.get("categories") if typ == "categorical" else None
            if typ == "categorical":
                if not categories:
                    continue
                categories = [str(category) for category in categories[:8]]
            try:
                selected_from_groups.append(
                    ExplicitFeatureSpec(
                        name=name,
                        type=typ,
                        categories=categories,
                        description=group.get("description") or name.replace("_", " "),
                        roles=roles,
                    )
                )
            except ValueError:
                continue
        return selected_from_groups

    grouped: Dict[str, List[ExplicitFeatureSpec]] = {}
    for specs in proposals_by_fold.values():
        seen_in_fold = set()
        for spec in specs:
            if required_role not in spec.roles:
                continue
            key = _normalize_feature_name(spec.name)
            if key in seen_in_fold:
                continue
            grouped.setdefault(key, []).append(spec)
            seen_in_fold.add(key)

    selected = []
    for key, specs in sorted(grouped.items()):
        if len(specs) < threshold:
            continue
        prototype = specs[0]
        roles = list(dict.fromkeys([role for spec in specs for role in spec.roles]))
        if required_role not in roles:
            roles.append(required_role)
        selected.append(
            ExplicitFeatureSpec(
                name=key,
                type=prototype.type,
                categories=prototype.categories,
                description=prototype.description,
                value_aliases=getattr(prototype, "value_aliases", None),
                roles=roles,
            )
        )
    return selected


def _proposals_by_fold_artifact(
    proposal_artifacts_by_fold: Dict[int, Sequence[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    return [
        {
            "fold": int(fold),
            "proposals": [dict(proposal) for proposal in proposal_artifacts_by_fold[fold]],
        }
        for fold in sorted(proposal_artifacts_by_fold)
    ]


def _validate_consensus_disambiguation_response(
    raw_response: Any,
    proposals_by_fold: Dict[int, Sequence[ExplicitFeatureSpec]],
    required_role: str,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Validate alias groups against actual per-fold proposals."""
    if not isinstance(raw_response, dict):
        return [], [f"response must be an object, got {type(raw_response).__name__}"]
    raw_groups = raw_response.get("groups", [])
    if not isinstance(raw_groups, list):
        return [], ["response.groups must be a list"]

    by_name: Dict[str, List[Tuple[int, ExplicitFeatureSpec]]] = {}
    for fold, specs in proposals_by_fold.items():
        seen_in_fold = set()
        for spec in specs:
            if required_role not in spec.roles:
                continue
            name = _normalize_feature_name(spec.name)
            if not name or name in seen_in_fold:
                continue
            by_name.setdefault(name, []).append((int(fold), spec))
            seen_in_fold.add(name)

    validated: List[Dict[str, Any]] = []
    errors: List[str] = []
    for group_idx, raw_group in enumerate(raw_groups, start=1):
        label = f"group {group_idx}"
        if not isinstance(raw_group, dict):
            errors.append(f"{label}: expected object, got {type(raw_group).__name__}")
            continue
        member_values = raw_group.get("member_names", [])
        if not isinstance(member_values, list):
            errors.append(f"{label}: member_names must be a list")
            continue
        member_names = [
            _normalize_feature_name(name) for name in member_values if _normalize_feature_name(name)
        ]
        member_names = list(dict.fromkeys(member_names))
        if not member_names:
            errors.append(f"{label}: no valid member_names")
            continue
        unknown_names = [name for name in member_names if name not in by_name]
        if unknown_names:
            errors.append(f"{label}: member_names were not proposed: {sorted(unknown_names)}")
            continue

        member_folds, fold_error = _normalize_member_folds(raw_group.get("member_folds"))
        if fold_error is not None:
            errors.append(f"{label}: {fold_error}")
            continue

        members: List[Tuple[int, str, ExplicitFeatureSpec]] = []
        for name in member_names:
            for fold, spec in by_name.get(name, []):
                if member_folds is None or fold in member_folds:
                    members.append((fold, name, spec))
        actual_folds = {fold for fold, _, _ in members}
        if member_folds is not None:
            missing_folds = sorted(set(member_folds) - actual_folds)
            if missing_folds:
                errors.append(f"{label}: no proposed member found in folds {missing_folds}")
                continue
        if len(actual_folds) < 2:
            errors.append(f"{label}: group must contain proposals from at least 2 distinct folds")
            continue

        member_specs = [spec for _, _, spec in members]
        member_types = {spec.type for spec in member_specs}
        if len(member_types) != 1:
            errors.append(f"{label}: grouped members have conflicting types")
            continue
        group_type = str(raw_group.get("type") or next(iter(member_types))).lower()
        if group_type not in VALID_TYPES:
            errors.append(f"{label}: invalid type {raw_group.get('type')!r}")
            continue
        if group_type not in member_types:
            errors.append(f"{label}: group type {group_type!r} conflicts with member proposals")
            continue

        categories = None
        if group_type == "categorical":
            category_signatures = {_category_signature(spec.categories) for spec in member_specs}
            if len(category_signatures) != 1:
                errors.append(f"{label}: grouped members have incompatible categories")
                continue
            category_signature = next(iter(category_signatures))
            group_categories = raw_group.get("categories")
            if group_categories:
                group_signature = _category_signature(group_categories)
                if group_signature != category_signature:
                    errors.append(f"{label}: group categories conflict with member proposals")
                    continue
                categories = [str(category) for category in group_categories[:8]]
            else:
                categories = list(next(spec.categories for spec in member_specs if spec.categories))
            if not categories:
                errors.append(f"{label}: categorical group has no categories")
                continue

        canonical_name = _normalize_feature_name(raw_group.get("canonical_name"))
        if canonical_name not in member_names:
            errors.append(f"{label}: canonical_name was not a proposed member; using first member")
            canonical_name = member_names[0]
        prototype = member_specs[0]
        roles = list(
            dict.fromkeys(
                role for spec in member_specs for role in spec.roles if role in VALID_ROLES
            )
        )
        if required_role not in roles:
            roles.append(required_role)
        validated.append(
            {
                "canonical_name": canonical_name,
                "member_names": member_names,
                "member_folds": sorted(actual_folds),
                "type": group_type,
                "categories": categories,
                "description": (
                    raw_group.get("description")
                    or prototype.description
                    or canonical_name.replace("_", " ")
                ),
                "rationale": raw_group.get("rationale"),
                "roles": roles,
                "members": [
                    {
                        "fold": int(fold),
                        "name": name,
                        "type": spec.type,
                        "categories": spec.categories,
                        "description": spec.description,
                        "roles": list(spec.roles),
                    }
                    for fold, name, spec in members
                ],
            }
        )
    return validated, errors


def _normalize_member_folds(raw_folds: Any) -> Tuple[Optional[set[int]], Optional[str]]:
    if raw_folds is None:
        return None, None
    if not isinstance(raw_folds, list):
        return None, "member_folds must be a list"
    folds: set[int] = set()
    for value in raw_folds:
        try:
            folds.add(int(value))
        except (TypeError, ValueError):
            return None, f"invalid member_folds value {value!r}"
    if not folds:
        return None, "member_folds must not be empty when provided"
    return folds, None


def _category_signature(categories: Optional[Sequence[Any]]) -> Tuple[str, ...]:
    return tuple(str(category).strip().lower() for category in (categories or []))


def _proposal_dicts_to_specs(
    raw_proposals: Any,
    required_role: str,
    max_specs: Optional[int] = None,
    excluded_feature_names: Optional[Sequence[str]] = None,
) -> List[ExplicitFeatureSpec]:
    raw_proposals = _proposal_list(raw_proposals)
    excluded = {
        _normalize_feature_name(name)
        for name in (excluded_feature_names or [])
        if _normalize_feature_name(name)
    }
    specs = []
    for proposal in raw_proposals or []:
        if not isinstance(proposal, dict):
            continue
        action = str(proposal.get("action", "add")).lower()
        if action not in {"add", "update_role"}:
            continue
        name = _normalize_feature_name(proposal.get("name", ""))
        if not name:
            continue
        if name in excluded:
            continue
        typ = str(proposal.get("type") or "categorical").lower()
        if typ not in VALID_TYPES:
            typ = "categorical"
        roles = [role for role in proposal.get("roles", []) if role in VALID_ROLES]
        if required_role not in roles:
            roles.append(required_role)
        categories = proposal.get("categories")
        if typ == "categorical":
            if not categories:
                categories = ["absent", "present"]
            categories = [str(cat) for cat in categories[:8]]
        else:
            categories = None
        try:
            specs.append(
                ExplicitFeatureSpec(
                    name=name,
                    type=typ,
                    categories=categories,
                    description=proposal.get("description") or name.replace("_", " "),
                    roles=roles,
                )
            )
            if max_specs is not None and len(specs) >= int(max_specs):
                break
        except ValueError:
            continue
    return specs


def _proposal_list(raw_proposals: Any) -> List[Dict[str, Any]]:
    if isinstance(raw_proposals, dict):
        raw_proposals = raw_proposals.get("proposals", [])
    if not isinstance(raw_proposals, list):
        return []
    return [proposal for proposal in raw_proposals if isinstance(proposal, dict)]


def _proposal_artifact_dicts(
    raw_proposals: Sequence[Dict[str, Any]],
    specs: Sequence[ExplicitFeatureSpec],
) -> List[Dict[str, Any]]:
    raw_by_name: Dict[str, List[Dict[str, Any]]] = {}
    for proposal in raw_proposals:
        name = _normalize_feature_name(proposal.get("name", ""))
        if name:
            raw_by_name.setdefault(name, []).append(proposal)

    artifacts: List[Dict[str, Any]] = []
    for spec in specs:
        row = _spec_to_dict(spec)
        raw = (raw_by_name.get(_normalize_feature_name(spec.name)) or [{}]).pop(0)
        for key in ["action", "rationale", "expected_signal"]:
            value = raw.get(key)
            if value is not None:
                row[key] = value
        artifacts.append(row)
    return artifacts


def _proposal_to_dict(proposal: AgenticFeatureProposal) -> Dict[str, Any]:
    return {
        "action": proposal.action,
        "name": proposal.name,
        "type": proposal.type,
        "categories": proposal.categories,
        "description": proposal.description,
        "roles": list(proposal.roles),
        "rationale": proposal.rationale,
        "expected_signal": proposal.expected_signal,
    }


def _feature_association_diagnostic(
    df: pd.DataFrame,
    spec: ExplicitFeatureSpec,
    config: AppliedInferenceConfig,
    existing_specs: Sequence[ExplicitFeatureSpec],
    alpha: float,
    min_n: int,
    min_non_missing: int,
) -> Dict[str, Any]:
    col = f"explicit_feat_{spec.name}"
    miss_col = f"{col}_missing"
    coverage = _feature_coverage(df, _normalize_feature_name(spec.name))
    if len(df) < min_n:
        return {
            "status": "skipped_insufficient_sample",
            "n": int(len(df)),
            "min_n": int(min_n),
            "coverage": float(coverage),
            "treatment_associated": True,
            "outcome_associated": True,
            "interaction_associated": False,
        }
    if col not in df.columns:
        return {
            "status": "missing_extracted_column",
            "coverage": 0.0,
            "treatment_associated": False,
            "outcome_associated": False,
            "interaction_associated": False,
        }
    missing = (
        df[miss_col].astype(bool).to_numpy()
        if miss_col in df.columns
        else df[col].isna().to_numpy()
    )
    non_missing_n = int((~missing).sum())
    if non_missing_n < min_non_missing:
        return {
            "status": "insufficient_non_missing",
            "n": int(len(df)),
            "coverage": float(coverage),
            "non_missing_n": non_missing_n,
            "min_non_missing": int(min_non_missing),
            "treatment_associated": False,
            "outcome_associated": False,
            "interaction_associated": False,
        }

    treatment_diag = _univariate_feature_target_association(
        df,
        spec,
        config.treatment_column,
        target_kind="binary",
    )
    outcome_diag = _univariate_feature_target_association(
        df,
        spec,
        config.outcome_column,
        target_kind=config.outcome_type,
    )
    interaction_diag = _treatment_interaction_association(
        df=df,
        spec=spec,
        config=config,
        existing_specs=existing_specs,
    )
    treatment_p = treatment_diag.get("p_value")
    outcome_p = outcome_diag.get("p_value")
    interaction_p = interaction_diag.get("p_value")
    return {
        "status": "ok",
        "n": int(len(df)),
        "coverage": float(coverage),
        "non_missing_n": non_missing_n,
        "alpha": float(alpha),
        "treatment_association": treatment_diag,
        "outcome_association": outcome_diag,
        "treatment_interaction": interaction_diag,
        "treatment_associated": _p_value_below(treatment_p, alpha),
        "outcome_associated": _p_value_below(outcome_p, alpha),
        "interaction_associated": _p_value_below(interaction_p, alpha),
    }


def _univariate_feature_target_association(
    df: pd.DataFrame,
    spec: ExplicitFeatureSpec,
    target_col: str,
    target_kind: str,
) -> Dict[str, Any]:
    col = f"explicit_feat_{spec.name}"
    miss_col = f"{col}_missing"
    if col not in df.columns or target_col not in df.columns:
        return {"status": "missing_column"}
    missing = df[miss_col].astype(bool) if miss_col in df.columns else df[col].isna()
    target = df[target_col]
    mask = (~missing) & df[col].notna() & target.notna()
    if int(mask.sum()) < 3:
        return {"status": "insufficient_rows", "n": int(mask.sum())}

    y_raw = target.loc[mask]
    x_raw = df.loc[mask, col]
    target_kind = "continuous" if target_kind == "continuous" else "binary"
    if target_kind == "binary":
        y_codes, uniques = pd.factorize(y_raw)
        if len(uniques) != 2:
            return {"status": "constant_target", "n": int(mask.sum())}
        y = y_codes.astype(float)
        if spec.type == "continuous":
            x = pd.to_numeric(x_raw, errors="coerce")
            finite = x.notna().to_numpy()
            if finite.sum() < 3 or len(np.unique(x[finite])) < 2:
                return {"status": "constant_feature", "n": int(finite.sum())}
            try:
                stat, p_value = stats.pointbiserialr(y[finite], x[finite].to_numpy(dtype=float))
            except Exception as exc:
                return {"status": "test_failed", "error": str(exc)}
            return {
                "status": "ok",
                "test": "point_biserial",
                "statistic": _finite_or_none(stat),
                "p_value": _finite_or_none(p_value),
                "n": int(finite.sum()),
            }
        table = pd.crosstab(x_raw.astype(str), y_raw.astype(str))
        if table.shape[0] < 2 or table.shape[1] != 2:
            return {"status": "constant_feature", "n": int(mask.sum())}
        try:
            chi2, p_value, dof, _ = stats.chi2_contingency(table.to_numpy())
        except Exception as exc:
            return {"status": "test_failed", "error": str(exc)}
        return {
            "status": "ok",
            "test": "chi_square",
            "statistic": _finite_or_none(chi2),
            "p_value": _finite_or_none(p_value),
            "dof": int(dof),
            "n": int(mask.sum()),
        }

    y = pd.to_numeric(y_raw, errors="coerce")
    if spec.type == "continuous":
        x = pd.to_numeric(x_raw, errors="coerce")
        finite = x.notna() & y.notna()
        if int(finite.sum()) < 3 or x[finite].nunique() < 2 or y[finite].nunique() < 2:
            return {"status": "constant_feature_or_target", "n": int(finite.sum())}
        try:
            stat, p_value = stats.pearsonr(
                x[finite].to_numpy(dtype=float), y[finite].to_numpy(dtype=float)
            )
        except Exception as exc:
            return {"status": "test_failed", "error": str(exc)}
        return {
            "status": "ok",
            "test": "pearson",
            "statistic": _finite_or_none(stat),
            "p_value": _finite_or_none(p_value),
            "n": int(finite.sum()),
        }

    groups = []
    for _, values in y.groupby(x_raw.astype(str)):
        values = values.dropna().to_numpy(dtype=float)
        if len(values) >= 2:
            groups.append(values)
    if len(groups) < 2:
        return {"status": "insufficient_groups", "n": int(mask.sum())}
    try:
        stat, p_value = stats.f_oneway(*groups)
    except Exception as exc:
        return {"status": "test_failed", "error": str(exc)}
    return {
        "status": "ok",
        "test": "anova",
        "statistic": _finite_or_none(stat),
        "p_value": _finite_or_none(p_value),
        "n": int(mask.sum()),
        "n_groups": int(len(groups)),
    }


def _treatment_interaction_association(
    df: pd.DataFrame,
    spec: ExplicitFeatureSpec,
    config: AppliedInferenceConfig,
    existing_specs: Sequence[ExplicitFeatureSpec],
) -> Dict[str, Any]:
    if config.outcome_type == "continuous":
        return _continuous_interaction_association(df, spec, config, existing_specs)
    if config.treatment_column not in df.columns or config.outcome_column not in df.columns:
        return {"status": "missing_target_column"}
    outcome = np.asarray(df[config.outcome_column].to_numpy(), dtype=float)
    treatment = np.asarray(df[config.treatment_column].to_numpy(), dtype=float)
    if (
        len(np.unique(outcome[~np.isnan(outcome)])) < 2
        or len(np.unique(treatment[~np.isnan(treatment)])) < 2
    ):
        return {"status": "constant_target_or_treatment"}

    current_confounders = [
        item for item in existing_specs if item.name != spec.name and "confounder" in item.roles
    ]
    _, w_matrix, _, _, _, _ = _build_features(df, current_confounders)
    candidate = ExplicitFeatureSpec(
        name=spec.name,
        type=spec.type,
        categories=spec.categories,
        description=spec.description,
        value_aliases=getattr(spec, "value_aliases", None),
        roles=["confounder"],
    )
    _, z_matrix, _, z_names, _, _ = _build_features(df, [candidate])
    w_matrix = _feature_matrix_or_empty(w_matrix, len(df))
    z_matrix = _feature_matrix_or_empty(z_matrix, len(df))
    if z_matrix.shape[1] == 0 or not _has_any_variation(z_matrix):
        return {"status": "constant_candidate", "candidate_feature_names": z_names}

    treatment_col = treatment.reshape(-1, 1)
    base_x = np.hstack([w_matrix, treatment_col, z_matrix])
    full_x = np.hstack([base_x, z_matrix * treatment_col])
    finite = (
        np.isfinite(outcome)
        & np.isfinite(treatment)
        & np.all(np.isfinite(base_x), axis=1)
        & np.all(np.isfinite(full_x), axis=1)
    )
    if int(finite.sum()) < 10:
        return {"status": "insufficient_finite_rows", "n": int(finite.sum())}
    try:
        p_value, lr_stat, dof = _binary_likelihood_ratio_p(
            base_x[finite],
            full_x[finite],
            outcome[finite],
            added_df=z_matrix.shape[1],
        )
    except Exception as exc:
        return {"status": "test_failed", "error": str(exc)}
    return {
        "status": "ok",
        "test": "logistic_likelihood_ratio",
        "p_value": _finite_or_none(p_value),
        "lr_statistic": _finite_or_none(lr_stat),
        "dof": int(dof),
        "candidate_feature_names": z_names,
        "n": int(finite.sum()),
    }


def _continuous_interaction_association(
    df: pd.DataFrame,
    spec: ExplicitFeatureSpec,
    config: AppliedInferenceConfig,
    existing_specs: Sequence[ExplicitFeatureSpec],
) -> Dict[str, Any]:
    outcome = np.asarray(df[config.outcome_column].to_numpy(), dtype=float)
    treatment = np.asarray(df[config.treatment_column].to_numpy(), dtype=float)
    current_confounders = [
        item for item in existing_specs if item.name != spec.name and "confounder" in item.roles
    ]
    _, w_matrix, _, _, _, _ = _build_features(df, current_confounders)
    candidate = ExplicitFeatureSpec(
        name=spec.name,
        type=spec.type,
        categories=spec.categories,
        description=spec.description,
        value_aliases=getattr(spec, "value_aliases", None),
        roles=["confounder"],
    )
    _, z_matrix, _, z_names, _, _ = _build_features(df, [candidate])
    w_matrix = _feature_matrix_or_empty(w_matrix, len(df))
    z_matrix = _feature_matrix_or_empty(z_matrix, len(df))
    if z_matrix.shape[1] == 0 or not _has_any_variation(z_matrix):
        return {"status": "constant_candidate", "candidate_feature_names": z_names}
    treatment_col = treatment.reshape(-1, 1)
    base_x = np.hstack([w_matrix, treatment_col, z_matrix])
    full_x = np.hstack([base_x, z_matrix * treatment_col])
    finite = (
        np.isfinite(outcome)
        & np.isfinite(treatment)
        & np.all(np.isfinite(base_x), axis=1)
        & np.all(np.isfinite(full_x), axis=1)
    )
    if int(finite.sum()) < 10:
        return {"status": "insufficient_finite_rows", "n": int(finite.sum())}
    try:
        p_value, f_stat, dof_num, dof_den = _linear_nested_f_test(
            base_x[finite],
            full_x[finite],
            outcome[finite],
            added_df=z_matrix.shape[1],
        )
    except Exception as exc:
        return {"status": "test_failed", "error": str(exc)}
    return {
        "status": "ok",
        "test": "linear_nested_f",
        "p_value": _finite_or_none(p_value),
        "f_statistic": _finite_or_none(f_stat),
        "dof_num": int(dof_num),
        "dof_den": int(dof_den),
        "candidate_feature_names": z_names,
        "n": int(finite.sum()),
    }


def _signal_feature_matrix(
    df: pd.DataFrame,
    specs: Sequence[ExplicitFeatureSpec],
) -> Tuple[Optional[np.ndarray], List[str]]:
    x_matrix, w_matrix, x_names, w_names, _, _ = _build_features(df, list(specs))
    matrix = _hstack_present(x_matrix, w_matrix)
    names = [*(x_names or []), *(w_names or [])]
    if matrix is None:
        return None, names
    return np.asarray(matrix, dtype=np.float32), names


def _cross_validated_boosted_signal_score(
    matrix: np.ndarray,
    target: np.ndarray,
    target_kind: str,
    folds: int,
    random_state: int,
) -> Dict[str, Any]:
    x = np.asarray(matrix, dtype=np.float32)
    y = np.asarray(target)
    finite = np.all(np.isfinite(x), axis=1) & pd.Series(y).notna().to_numpy()
    x = x[finite]
    y = y[finite]
    if len(y) < 10:
        return {"status": "insufficient_rows", "n": int(len(y))}
    if target_kind != "continuous":
        y_codes, uniques = pd.factorize(y)
        if len(uniques) != 2:
            return {"status": "constant_target", "n": int(len(y))}
        y_binary = y_codes.astype(int)
        class_counts = np.bincount(y_binary)
        n_splits = min(int(folds), int(class_counts.min()))
        if n_splits < 2:
            return {
                "status": "insufficient_class_counts",
                "n": int(len(y)),
                "class_counts": class_counts.tolist(),
            }
        preds = np.full(len(y_binary), np.nan, dtype=float)
        model_name = None
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(x, y_binary)):
            pred, model_name = _fit_predict_boosted_classifier(
                x[train_idx],
                y_binary[train_idx],
                x[test_idx],
                random_state=random_state + fold_idx,
            )
            preds[test_idx] = pred
        mask = np.isfinite(preds)
        return {
            "status": "ok",
            "target_kind": "binary",
            "metric": "auroc",
            "score": _safe_roc_auc(y_binary[mask], preds[mask]),
            "model": model_name,
            "n": int(mask.sum()),
            "folds": int(n_splits),
        }

    y_cont = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(y_cont)
    x = x[finite]
    y_cont = y_cont[finite]
    if len(y_cont) < 10 or np.std(y_cont) == 0:
        return {"status": "insufficient_or_constant_target", "n": int(len(y_cont))}
    n_splits = min(int(folds), len(y_cont))
    if n_splits < 2:
        return {"status": "insufficient_rows", "n": int(len(y_cont))}
    preds = np.full(len(y_cont), np.nan, dtype=float)
    model_name = None
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(x)):
        pred, model_name = _fit_predict_boosted_regressor(
            x[train_idx],
            y_cont[train_idx],
            x[test_idx],
            random_state=random_state + fold_idx,
        )
        preds[test_idx] = pred
    mask = np.isfinite(preds)
    return {
        "status": "ok",
        "target_kind": "continuous",
        "metric": "r2",
        "score": float(r2_score(y_cont[mask], preds[mask])),
        "model": model_name,
        "n": int(mask.sum()),
        "folds": int(n_splits),
    }


def _fit_predict_boosted_classifier(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    random_state: int,
) -> Tuple[np.ndarray, str]:
    try:
        from xgboost import XGBClassifier  # type: ignore

        model = XGBClassifier(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=1,
            verbosity=0,
        )
        model.fit(train_x, train_y)
        return model.predict_proba(test_x)[:, 1], "xgboost.XGBClassifier"
    except Exception:
        model = HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.05,
            max_leaf_nodes=15,
            random_state=random_state,
        )
        model.fit(train_x, train_y)
        return model.predict_proba(test_x)[:, 1], "sklearn.HistGradientBoostingClassifier"


def _fit_predict_boosted_regressor(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    random_state: int,
) -> Tuple[np.ndarray, str]:
    try:
        from xgboost import XGBRegressor  # type: ignore

        model = XGBRegressor(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=1,
            verbosity=0,
        )
        model.fit(train_x, train_y)
        return model.predict(test_x), "xgboost.XGBRegressor"
    except Exception:
        model = HistGradientBoostingRegressor(
            max_iter=120,
            learning_rate=0.05,
            max_leaf_nodes=15,
            random_state=random_state,
        )
        model.fit(train_x, train_y)
        return model.predict(test_x), "sklearn.HistGradientBoostingRegressor"


def _score_meets_signal_threshold(score: Dict[str, Any], threshold: float) -> bool:
    if score.get("status") == "skipped_insufficient_sample":
        return True
    value = score.get("score")
    if value is None or not np.isfinite(float(value)):
        return False
    if score.get("metric") == "auroc":
        return float(value) >= float(threshold)
    return float(value) > 0.0


def _binary_likelihood_ratio_p(
    base_x: np.ndarray,
    full_x: np.ndarray,
    y: np.ndarray,
    added_df: int,
) -> Tuple[float, float, int]:
    y_codes, uniques = pd.factorize(y)
    if len(uniques) != 2:
        raise ValueError("binary likelihood ratio requires two outcome classes")
    y_binary = y_codes.astype(int)
    base_model = LogisticRegression(max_iter=1000, solver="lbfgs")
    full_model = LogisticRegression(max_iter=1000, solver="lbfgs")
    base_model.fit(_ensure_model_columns(base_x), y_binary)
    full_model.fit(_ensure_model_columns(full_x), y_binary)
    base_pred = np.clip(
        base_model.predict_proba(_ensure_model_columns(base_x))[:, 1], 1e-6, 1 - 1e-6
    )
    full_pred = np.clip(
        full_model.predict_proba(_ensure_model_columns(full_x))[:, 1], 1e-6, 1 - 1e-6
    )
    base_ll = -log_loss(y_binary, base_pred, labels=[0, 1], normalize=False)
    full_ll = -log_loss(y_binary, full_pred, labels=[0, 1], normalize=False)
    lr_stat = max(0.0, 2.0 * (float(full_ll) - float(base_ll)))
    dof = max(1, int(added_df))
    return float(stats.chi2.sf(lr_stat, dof)), float(lr_stat), dof


def _linear_nested_f_test(
    base_x: np.ndarray,
    full_x: np.ndarray,
    y: np.ndarray,
    added_df: int,
) -> Tuple[float, float, int, int]:
    base_x = _ensure_model_columns(base_x)
    full_x = _ensure_model_columns(full_x)
    base_model = LinearRegression()
    full_model = LinearRegression()
    base_model.fit(base_x, y)
    full_model.fit(full_x, y)
    base_resid = y - base_model.predict(base_x)
    full_resid = y - full_model.predict(full_x)
    rss_base = float(np.sum(base_resid**2))
    rss_full = float(np.sum(full_resid**2))
    dof_num = max(1, int(added_df))
    dof_den = max(1, int(len(y) - full_x.shape[1] - 1))
    f_stat = max(0.0, ((rss_base - rss_full) / dof_num) / max(rss_full / dof_den, 1e-12))
    return float(stats.f.sf(f_stat, dof_num, dof_den)), float(f_stat), dof_num, dof_den


def _feature_matrix_or_empty(matrix: Optional[np.ndarray], n_rows: int) -> np.ndarray:
    if matrix is None:
        return np.zeros((n_rows, 0), dtype=np.float32)
    return np.asarray(matrix, dtype=np.float32)


def _ensure_model_columns(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.shape[1] == 0:
        return np.zeros((x.shape[0], 1), dtype=np.float64)
    return x


def _has_any_variation(matrix: np.ndarray) -> bool:
    matrix = np.asarray(matrix, dtype=float)
    return bool(matrix.size and np.any(np.nanstd(matrix, axis=0) > 1e-12))


def _p_value_below(value: Any, alpha: float) -> bool:
    try:
        return bool(value is not None and np.isfinite(float(value)) and float(value) < alpha)
    except (TypeError, ValueError):
        return False


def _finite_or_none(value: Any) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _fit_predict_propensity(
    train_x: np.ndarray,
    train_t: np.ndarray,
    test_x: np.ndarray,
    cf_config: ExplicitFeatureForestConfig,
    random_state: int,
) -> np.ndarray:
    if len(np.unique(train_t)) < 2:
        return np.full(len(test_x), float(np.mean(train_t)))
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
        return np.full(len(test_x), float(np.mean(train_y)))
    model = RandomForestClassifier(
        n_estimators=max(50, cf_config.n_estimators // 2),
        max_depth=cf_config.max_depth,
        min_samples_leaf=cf_config.min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(train_x, train_y)
    return model.predict_proba(test_x)[:, 1]


def _run_crossfit_fold_tasks(
    run_fold,
    split_items,
    n_jobs: int,
    device_context=None,
) -> List[Dict[str, Any]]:
    def call_fold(fold, fit_pos, heldout_pos):
        if device_context is None:
            return run_fold(fold, fit_pos, heldout_pos)
        with device_context(fold):
            return run_fold(fold, fit_pos, heldout_pos)

    if n_jobs <= 1:
        return [
            call_fold(fold, fit_pos, heldout_pos) for fold, (fit_pos, heldout_pos) in split_items
        ]
    with ThreadPoolExecutor(
        max_workers=int(n_jobs),
        thread_name_prefix="avf-fold",
    ) as executor:
        futures = [
            executor.submit(call_fold, fold, fit_pos, heldout_pos)
            for fold, (fit_pos, heldout_pos) in split_items
        ]
        return [future.result() for future in futures]


def _bounded_fold_count(requested: int, n: int) -> int:
    if n < 2:
        raise ValueError("At least two rows are required for cross-fitting")
    return max(2, min(int(requested), int(n)))


def _is_cuda_oom(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "cuda" in message and "out of memory" in message


def _normalize_feature_name(name: Any) -> str:
    value = str(name or "").strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def _feature_coverage(df: pd.DataFrame, name: str) -> float:
    col = f"explicit_feat_{name}"
    miss_col = f"{col}_missing"
    if col not in df.columns:
        return 0.0
    if miss_col in df.columns:
        missing = df[miss_col].astype(bool)
    else:
        missing = df[col].isna()
    return float(1.0 - missing.mean())


def _spec_to_dict(spec: ExplicitFeatureSpec) -> Dict[str, Any]:
    return {
        "name": spec.name,
        "type": spec.type,
        "categories": spec.categories,
        "description": spec.description,
        "value_aliases": getattr(spec, "value_aliases", None),
        "roles": list(spec.roles),
    }


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return None


def _format_optional_metric(value: Optional[float]) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{float(value):.4f}"


def _safe_corr(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, default=_json_default) + "\n")


def _write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        df.to_parquet(tmp_path, index=False)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _write_json_atomic(data: Dict[str, Any], path: Path) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2, default=_json_default)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _hash_numeric_array(values: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(str(arr.shape).encode("utf-8"))
    digest.update(arr.tobytes())
    return digest.hexdigest()


def _parse_top_token_spans(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if not isinstance(value, str) or not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def _attention_row_has_usable_text(row: Dict[str, Any]) -> bool:
    for key in ["chunk_text", "highlighted_chunk_text", "attended_token_summary"]:
        value = row.get(key)
        if isinstance(value, str) and re.search(r"[A-Za-z0-9]", value):
            return True
    for span in _parse_top_token_spans(row.get("top_token_spans_json")):
        text = span.get("text")
        if isinstance(text, str) and re.search(r"[A-Za-z0-9]", text):
            return True
    return False


def _compact_token_spans(spans: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    compact: List[Dict[str, Any]] = []
    seen = set()
    for span in spans:
        text = _normalize_context_text(str(span.get("text", "")).strip())
        if not text or text in seen:
            continue
        seen.add(text)
        item: Dict[str, Any] = {"text": text}
        focus = str(span.get("focus_token", "")).strip()
        if focus and focus != text:
            item["focus_token"] = _normalize_context_text(focus)
        if "salience" in span:
            item["salience"] = _round_context_float(span["salience"], ndigits=5)
        compact.append(item)
    return compact


def _attention_evidence_snippet(
    chunk_text: Any,
    spans: Sequence[Dict[str, Any]],
    highlighted_chunk_text: Any = None,
) -> str:
    chunk = _normalize_context_text(chunk_text)
    if chunk:
        return chunk

    highlighted = _normalize_context_text(highlighted_chunk_text)
    return highlighted


def _normalize_context_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip()


def _round_context_float(value: Any, ndigits: int = 6) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(numeric):
        return 0.0
    return round(numeric, ndigits)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return str(value)


def _scrub_context(context: Dict[str, Any]) -> Dict[str, Any]:
    copied = dict(context)
    if "attention_evidence" in copied:
        copied["attention_evidence"] = [
            {
                key: value
                for key, value in row.items()
                if key not in {"chunk_text", "source_chunk_text"}
            }
            for row in copied.get("attention_evidence", [])
        ]
    return copied
