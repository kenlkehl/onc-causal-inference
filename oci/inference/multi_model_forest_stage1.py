"""Stage 1 text-model training for the researcher all-evidence workflow."""

from __future__ import annotations

import copy
import logging
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, TypeVar

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.metrics import brier_score_loss, log_loss, mean_squared_error
from sklearn.model_selection import KFold

from ..config import (
    AppliedInferenceConfig,
    BoWViewConfig,
    MultiModelForestConfig,
    resolve_effect_objective,
)
from ..utils.calibration import clip_probability
from .causal_modeling_support import safe_roc_auc as _safe_roc_auc
from .embedding_contrast_discovery import EmbeddingContrastEvidenceGenerator
from .discovery_randomness import derive_discovery_seed
from .htr_evidence_provider import HTREvidenceProvider
from .htr_trainer import (
    EffectNet,
    NuisanceNet,
    _metadata_value,
    cross_fitted_calibrators,
    effect_objective_name,
    r_pseudo_outcome,
)
from .stage1_modeling_support import (
    _agentic_discovery_handoff_row,
    _align_htr_prediction_frame,
    _htr_effect_metrics,
    _htr_nuisance_metrics,
    _multi_view_importance,
    _normalize_texts,
    _top_phrase_feature_rows,
)
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
    _model_feature_scores_are_signed,
    _top_feature_rows,
)
from .multi_model_pair_uplift import (
    BINARY_PROGNOSTIC_SOURCE,
    MATCHED_PAIR_CHECKPOINT_VERSION,
    PROPENSITY_SOURCE,
    MatchedPairLaneResult,
    MatchedPairSettings,
    crossfit_prognostic_score,
    distinct_vectorizer_groups,
    matched_pair_seed,
    run_matched_pair_lane,
)
from .stage1_architecture_artifacts import (
    EMBEDDING_DIRECTION_SCORE_SOURCES,
    EMBEDDING_DIRECTION_SCORE_STATISTICS,
    embedding_direction_score_column,
)
from .stage1_architectures import (
    BOW_NUISANCE,
    BOW_R_LOSS,
    EMBEDDING_CLUSTERED,
    EMBEDDING_WHOLE_COHORT,
    HTR_NEURAL,
    MATCHED_PAIR_UPLIFT,
)
from .stage1_checkpointing import Stage1CheckpointStore, row_id_identity
from .stage1_timing import (
    EMBEDDING_ARCHITECTURES,
    SHARED_ATTRIBUTION,
    TEXT_MODEL_ARCHITECTURES,
    NullTimer,
)

logger = logging.getLogger(__name__)

_CheckpointValue = TypeVar("_CheckpointValue")


@dataclass
class _FeatureBundle:
    prediction_frames: List[pd.DataFrame]
    metrics: Dict[str, Any]
    handoff_evidence: Optional[Dict[str, Any]]


def _feature_bundle_checkpoint_payload(bundle: _FeatureBundle) -> Dict[str, Any]:
    return {
        "prediction_frames": bundle.prediction_frames,
        "metrics": bundle.metrics,
        "handoff_evidence": bundle.handoff_evidence,
    }


def _feature_bundle_from_checkpoint(payload: Mapping[str, Any]) -> _FeatureBundle:
    return _FeatureBundle(
        prediction_frames=list(payload["prediction_frames"]),
        metrics=dict(payload["metrics"]),
        handoff_evidence=(
            None
            if payload.get("handoff_evidence") is None
            else dict(payload["handoff_evidence"])
        ),
    )


def _validate_train_test_prediction_tuple(
    result: Any,
    train_rows: int,
    test_rows: int,
    label: str,
) -> None:
    if not isinstance(result, (list, tuple)) or len(result) != 2:
        raise ValueError(f"{label} checkpoint must contain fit and held-out predictions")
    if len(np.asarray(result[0])) != int(train_rows):
        raise ValueError(f"{label} checkpoint fit row count changed")
    if len(np.asarray(result[1])) != int(test_rows):
        raise ValueError(f"{label} checkpoint held-out row count changed")


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
    if not isinstance(train_predictions, pd.DataFrame) or len(train_predictions) != train_rows:
        raise ValueError("nuisance ensemble checkpoint train prediction rows changed")


def _validate_mapping_checkpoint(value: Any, label: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} checkpoint is not a mapping")


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


def run_multi_model_forest_handoff_context(
    *,
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    context: Mapping[str, Any],
    handoff_dir: Path,
    device: Any,
    num_workers: int,
    checkpoint_store: Optional[Stage1CheckpointStore] = None,
    timer: Any = None,
) -> Dict[str, Any]:
    """Fit all enabled Stage 1 text architectures in one exact discovery context.

    Evidence is fitted only on ``context["train_idx"]``; ``context["heldout_idx"]``
    rows are prediction targets.  The returned row is the context's handoff
    evidence.  ``timer`` (reporting only) receives one lap per section.
    """

    context_config = copy.deepcopy(config)
    context_config.architecture.model_type = "multi_model_forest"
    train_idx = np.asarray(context["train_idx"], dtype=int)
    heldout_idx = np.asarray(context["heldout_idx"], dtype=int)
    with _serial_torch_worker_environment():
        runner = MultiModelForestStage1Runner(
            dataset=dataset,
            config=context_config,
            output_path=(
                Path(handoff_dir)
                / "worker_artifacts"
                / f"fold_{int(context['fold_key']):06d}"
                / "predictions.parquet"
            ),
            device=torch.device(device),
            num_workers=max(1, int(num_workers)),
            checkpoint_store=checkpoint_store,
            timer=timer,
        )
        logger.info(
            "Precomputing handoff context fold_key=%s scope=%s rows=%s device=%s",
            context["fold_key"],
            context["scope"],
            len(train_idx),
            device,
        )
        return runner.build_discovery_handoff_row(
            train_idx=train_idx,
            heldout_idx=heldout_idx,
            fold_key=int(context["fold_key"]),
            outer_fold=int(context["outer_fold"]),
            scope=str(context["scope"]),
            inner_fold=context.get("inner_fold"),
            heldout_rows=context.get("heldout_rows"),
        )


@contextmanager
def _serial_torch_worker_environment():
    """Constrain nested numerical pools inside one Stage 1 context worker."""

    from threadpoolctl import threadpool_limits

    env_keys = [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]
    previous = {key: os.environ.get(key) for key in env_keys}
    for key in env_keys:
        os.environ[key] = "1"
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


class MultiModelForestStage1HTRProvider(HTREvidenceProvider):
    """HTR inner-fold ensembles with full outer-train -> outer-test predictions."""

    @contextmanager
    def _temporary_effect_objective(self, objective: str):
        runner = self._runner
        if runner is None:
            raise RuntimeError("HTR runner has not been initialized")
        # Binary outcomes require DINA and DINA requires binary outcomes; a
        # mismatched variant raises instead of being silently replaced.
        resolved = resolve_effect_objective(objective, outcome_type=runner.config.outcome_type)
        previous = runner.avf_config.effect_objective
        runner.avf_config.effect_objective = resolved
        try:
            yield
        finally:
            runner.avf_config.effect_objective = previous

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
                model = NuisanceNet(
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
                heldout = train_df.iloc[heldout_pos]
                # Calibration happens after every fold has produced its raw
                # out-of-fold predictions (see cross_fitted_calibrators); the
                # model's in-sample predictions on fit_pos are never used.
                e_raw, m_raw = runner._predict_nuisance_model(model, heldout)
                e_test_raw, m_test_raw = runner._predict_nuisance_model(model, test_df)
                fold_attention = runner._attention_evidence(
                    model.extractor,
                    heldout,
                    fold=fold,
                    outer_fold=outer_fold,
                    stage="nuisance",
                    extra={"e_hat_raw": e_raw, "m_hat_raw": m_raw},
                )
                return {
                    "fold": int(fold),
                    "heldout_pos": heldout_pos,
                    "e_hat_raw": e_raw,
                    "m_hat_raw": m_raw,
                    "test_e_hat_raw": e_test_raw,
                    "test_m_hat_raw": m_test_raw,
                    "attention": fold_attention,
                }
            finally:
                if model is not None:
                    runner._cleanup_model(model)

        def run_fold(fold: int, fit_pos: np.ndarray, heldout_pos: np.ndarray):
            store = self.checkpoint_store
            if store is None:
                return compute_fold(fold, fit_pos, heldout_pos)
            fit_pos = np.asarray(fit_pos, dtype=int)
            heldout_pos = np.asarray(heldout_pos, dtype=int)

            def validate(result: Mapping[str, Any]) -> None:
                if len(np.asarray(result["heldout_pos"])) != len(heldout_pos):
                    raise ValueError("HTR nuisance checkpoint held-out row count changed")
                for name in ("e_hat_raw", "m_hat_raw"):
                    if len(np.asarray(result[name])) != len(heldout_pos):
                        raise ValueError(f"HTR nuisance checkpoint {name} row count changed")
                for name in ("test_e_hat_raw", "test_m_hat_raw"):
                    if len(np.asarray(result[name])) != len(test_df):
                        raise ValueError(f"HTR nuisance checkpoint {name} row count changed")

            # Torch RNG is process-global.  Serial checkpointed leaves make the
            # per-leaf seed deterministic without retaining epoch/model state.
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

        fold_results = [
            run_fold(int(fold), np.asarray(fit_pos), np.asarray(heldout_pos))
            for fold, (fit_pos, heldout_pos) in split_items
        ]
        # Out-of-fold calibration: fold k's calibrator is fit on the raw
        # predictions that the other folds' models made for fold k's fit rows.
        e_raw_oof = np.full(len(train_df), np.nan, dtype=float)
        m_raw_oof = np.full(len(train_df), np.nan, dtype=float)
        fold_of_row = np.zeros(len(train_df), dtype=int)
        for result in fold_results:
            heldout_pos = np.asarray(result["heldout_pos"], dtype=int)
            e_raw_oof[heldout_pos] = np.asarray(result["e_hat_raw"], dtype=float)
            m_raw_oof[heldout_pos] = np.asarray(result["m_hat_raw"], dtype=float)
            fold_of_row[heldout_pos] = int(result["fold"])
        t_all = train_df[runner.config.treatment_column].to_numpy(dtype=float)
        y_all = train_df[runner.config.outcome_column].to_numpy(dtype=float)
        calibration_method = runner.avf_config.nuisance_calibration
        propensity_calibrators = cross_fitted_calibrators(
            e_raw_oof,
            t_all,
            fold_of_row,
            method=calibration_method,
        )
        outcome_calibrators = (
            None
            if runner.config.outcome_type == "continuous"
            else cross_fitted_calibrators(
                m_raw_oof,
                y_all,
                fold_of_row,
                method=calibration_method,
            )
        )
        attention_rows: List[Dict[str, Any]] = []
        test_e = []
        test_m = []
        for result in fold_results:
            fold = int(result["fold"])
            heldout_pos = np.asarray(result["heldout_pos"], dtype=int)
            e_hat = propensity_calibrators[fold].transform(result["e_hat_raw"])
            test_e_hat = propensity_calibrators[fold].transform(result["test_e_hat_raw"])
            if outcome_calibrators is None:
                m_hat = np.asarray(result["m_hat_raw"], dtype=float)
                test_m_hat = np.asarray(result["test_m_hat_raw"], dtype=float)
            else:
                m_hat = outcome_calibrators[fold].transform(result["m_hat_raw"])
                test_m_hat = outcome_calibrators[fold].transform(result["test_m_hat_raw"])
            y_resid = y_all[heldout_pos] - m_hat
            t_resid = t_all[heldout_pos] - e_hat
            predictions.loc[heldout_pos, "e_hat"] = e_hat
            predictions.loc[heldout_pos, "e_hat_raw"] = result["e_hat_raw"]
            predictions.loc[heldout_pos, "m_hat"] = m_hat
            predictions.loc[heldout_pos, "m_hat_raw"] = result["m_hat_raw"]
            predictions.loc[heldout_pos, "y_residual"] = y_resid
            predictions.loc[heldout_pos, "t_residual"] = t_resid
            predictions.loc[heldout_pos, "r_pseudo_outcome"] = r_pseudo_outcome(
                y_resid,
                t_resid,
            )
            predictions.loc[heldout_pos, "r_loss_at_zero_tau"] = y_resid**2
            predictions.loc[heldout_pos, "nuisance_fold"] = fold
            calibrated_by_row = {
                int(row_id): {
                    "e_hat": e_value,
                    "m_hat": m_value,
                    "y_residual": y_value,
                    "t_residual": t_value,
                }
                for row_id, e_value, m_value, y_value, t_value in zip(
                    train_df["_oci_row_id"].to_numpy()[heldout_pos],
                    e_hat,
                    m_hat,
                    y_resid,
                    t_resid,
                )
            }
            for record in result["attention"]:
                values = calibrated_by_row.get(int(record["row_id"]))
                if values is not None:
                    record.update(
                        {
                            key: _metadata_value(value, 0)
                            for key, value in values.items()
                        }
                    )
            attention_rows.extend(result["attention"])
            test_e.append(np.asarray(test_e_hat, dtype=float))
            test_m.append(np.asarray(test_m_hat, dtype=float))
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
        }

    def fit_effect_variant_inner_ensemble_predict(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        nuisance_predictions: pd.DataFrame,
        outer_fold: int,
        *,
        effect_objective: str,
    ) -> Dict[str, Any]:
        runner = self._ensure_runner(train_df)
        folds = _bounded_fold_count(runner.avf_config.effect_folds, len(train_df))
        with self._temporary_effect_objective(effect_objective):
            effect_objective = effect_objective_name(runner.avf_config)
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
            pseudo_outcome = r_pseudo_outcome(y_resid, t_resid)
            r_stage_min = float(getattr(runner.avf_config, "r_stage_min_propensity", 0.0))
            r_stage_max = float(getattr(runner.avf_config, "r_stage_max_propensity", 1.0))
            train_eligible = np.isfinite(e) & (e >= r_stage_min) & (e <= r_stage_max)
            if effect_objective == "pseudo_outcome_mse":
                train_eligible = train_eligible & np.isfinite(pseudo_outcome)
            r_df["tau_hat_r_stage"] = np.nan
            r_df["tau_logit_modifier"] = np.nan
            r_df["r_loss"] = np.nan
            r_df["effect_loss"] = np.nan
            r_df["effect_loss_at_zero_tau"] = y_resid**2
            r_df["effect_fold"] = np.nan
            r_df["r_stage_train_eligible"] = train_eligible
            r_df["effect_objective"] = effect_objective
            r_df["r_pseudo_outcome"] = pseudo_outcome

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
                    model = EffectNet(
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
                    if effect_objective == "dina":
                        from .stage1_dina import neural_predictions
                        from ..models import dina

                        dp = neural_predictions(runner, model, heldout, raw_effect)
                        dt = neural_predictions(runner, model, test_df, raw_test)
                        tau_hat, test_tau, tau_logit_modifier = dp["tau"], dt["tau"], dp["delta"]
                        heldout_effect_loss = dina.loss(
                            y[heldout_pos], t[heldout_pos], dp["a"], dp["nu"], dp["delta"]
                        )
                        effect_loss_at_zero_tau = dina.loss(
                            y[heldout_pos], t[heldout_pos], dp["a"], dp["nu"], 0.0
                        )
                    else:
                        tau_hat = raw_effect
                        test_tau = raw_test
                        tau_logit_modifier = np.full(len(heldout_pos), np.nan)
                        heldout_effect_loss = (
                            (tau_hat - pseudo_outcome[heldout_pos]) ** 2
                            if effect_objective == "pseudo_outcome_mse"
                            else (y_resid[heldout_pos] - tau_hat * t_resid[heldout_pos]) ** 2
                        )
                        effect_loss_at_zero_tau = r_df.iloc[heldout_pos][
                            "effect_loss_at_zero_tau"
                        ].to_numpy()
                    heldout_r_loss = (y_resid[heldout_pos] - tau_hat * t_resid[heldout_pos]) ** 2
                    fold_attention = runner._attention_evidence(
                        model.extractor,
                        heldout,
                        fold=fold,
                        outer_fold=outer_fold,
                        stage="effect_modifier",
                        extra={
                            "tau_hat_r_stage": tau_hat,
                            "tau_logit_modifier": tau_logit_modifier,
                            "r_pseudo_outcome": pseudo_outcome[heldout_pos],
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
                        "effect_loss_at_zero_tau": effect_loss_at_zero_tau,
                        "test_tau": test_tau,
                        "attention": fold_attention,
                    }
                finally:
                    if model is not None:
                        runner._cleanup_model(model)

            def run_fold(fold: int, fit_pos: np.ndarray, heldout_pos: np.ndarray):
                store = self.checkpoint_store
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

            fold_results = [
                run_fold(int(fold), np.asarray(fit_pos), np.asarray(heldout_pos))
                for fold, (fit_pos, heldout_pos) in split_items
            ]
            attention_rows: List[Dict[str, Any]] = []
            test_tau_predictions = []
            for result in fold_results:
                heldout_pos = result["heldout_pos"]
                r_df.loc[heldout_pos, "tau_hat_r_stage"] = result["tau_hat"]
                r_df.loc[heldout_pos, "tau_logit_modifier"] = result["tau_logit_modifier"]
                r_df.loc[heldout_pos, "r_loss"] = result["r_loss"]
                r_df.loc[heldout_pos, "effect_loss"] = result["effect_loss"]
                r_df.loc[heldout_pos, "effect_loss_at_zero_tau"] = result["effect_loss_at_zero_tau"]
                r_df.loc[heldout_pos, "effect_fold"] = result["fold"]
                attention_rows.extend(result["attention"])
                test_tau_predictions.append(np.asarray(result["test_tau"], dtype=float))
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
            }


class MultiModelForestStage1Runner:
    """Fit every enabled Stage 1 text architecture for one exact context."""

    def __init__(
        self,
        dataset: pd.DataFrame,
        config: AppliedInferenceConfig,
        output_path: Path,
        device: Optional[Any] = None,
        num_workers: int = 1,
        checkpoint_store: Optional[Stage1CheckpointStore] = None,
        timer: Any = None,
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
        self.num_workers = 1 if num_workers is None else int(num_workers)
        self.checkpoint_store = checkpoint_store
        self.nn_config: MultiModelForestConfig = getattr(
            config.architecture,
            "multi_model_forest",
            MultiModelForestConfig(),
        )
        self.embedding_evidence_generator: Optional[EmbeddingContrastEvidenceGenerator] = None
        self._default_htr_provider: Optional[MultiModelForestStage1HTRProvider] = None
        # Reporting-only wall-time sections; never part of a checkpoint identity.
        self.timer = timer if timer is not None else NullTimer()

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
        store = self.checkpoint_store
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

        Evidence is fitted only from ``train_idx``; held-out rows are
        prediction targets.
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
        self.timer.lap(
            "prediction_artifacts",
            attribution=SHARED_ATTRIBUTION,
            consumers=TEXT_MODEL_ARCHITECTURES,
        )
        result = copy.deepcopy(bundle.handoff_evidence or {})
        metrics = dict(result.get("metrics") or {})
        metrics.update(bundle.metrics)
        metrics["handoff_fold_key"] = int(fold_key)
        metrics["handoff_fit_rows"] = int(len(train_df))
        metrics["handoff_heldout_rows"] = int(len(heldout_df))
        result["metrics"] = metrics
        row = _agentic_discovery_handoff_row(
            result,
            fold_key=int(fold_key),
            outer_fold=int(outer_fold),
            scope=str(scope),
            n_rows=len(train_df),
            inner_fold=inner_fold,
            heldout_rows=(len(heldout_df) if heldout_rows is None else int(heldout_rows)),
        )
        # Exact row provenance, as TF-IDF and neural-query contexts record, so
        # Stage 2 verifies this context against split_provenance.jsonl by row
        # identity rather than by row counts.
        row["fit_row_ids"] = [int(value) for value in train_df["_oci_row_id"]]
        row["heldout_row_ids"] = [int(value) for value in heldout_df["_oci_row_id"]]
        return row

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
            required = {"prediction_frames", "metrics", "handoff_evidence"}
            missing = sorted(required - set(payload))
            if missing:
                raise ValueError(f"feature-bundle checkpoint is missing fields: {missing}")

        payload = self._checkpointed_value(
            "assembly/feature_bundle",
            compute,
            train_df=train_df,
            test_df=test_df,
            # 4: embedding row scores come from the handed-off directions, and
            # matched_pair_uplift is the BoW conditional-logit lane.
            parameters={"outer_fold": int(outer_fold), "format_version": 4},
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
        prediction_frames: List[pd.DataFrame] = []
        metrics: Dict[str, Any] = {}
        nuisance_train: List[Tuple[str, np.ndarray, np.ndarray]] = []
        nuisance_test: List[Tuple[str, np.ndarray, np.ndarray]] = []
        bow_nuisance_by_view: List[Dict[str, Any]] = []
        bow_view_results: List[Dict[str, Any]] = []
        htr_evidence: Dict[str, Any] = {}
        nuisance_checkpoint_units: List[str] = []

        if self._bow_enabled():
            for view_index, view in enumerate(self.nn_config.bow_views):
                e_train, e_test = self._checkpointed_value(
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
                if str(self.config.outcome_type).lower() == "continuous":
                    m_train, m_test = self._checkpointed_value(
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
                else:
                    m_train, m_test = self._checkpointed_value(
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
                nuisance_train.append((view.name, e_train, m_train))
                nuisance_test.append((view.name, e_test, m_test))
                bow_nuisance_by_view.append(
                    {
                        "view": view,
                        "view_index": int(view_index),
                        "e_hat": e_train,
                        "m_hat": m_train,
                    }
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
            self.timer.lap(
                "bow_nuisance_fits",
                attribution=SHARED_ATTRIBUTION,
                consumers=TEXT_MODEL_ARCHITECTURES,
            )

        if self._htr_enabled():
            htr_bundle = self._htr_provider().fit_nuisance_inner_ensemble_predict(
                train_df,
                test_df,
                outer_fold,
            )
            htr_train_result = htr_bundle["train"]
            htr_test_predictions = htr_bundle["test_predictions"]
            htr_nuisance_folds = _bounded_fold_count(
                self.config.architecture.agentic_attention_variable_forest.nuisance_folds,
                len(train_df),
            )
            nuisance_checkpoint_units.extend(
                f"htr/nuisance/fold_{fold:03d}"
                for fold in range(1, htr_nuisance_folds + 1)
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
            nuisance_train.append(("htr_nuisance", htr_e_train, htr_m_train))
            nuisance_test.append(("htr_nuisance", htr_e_test, htr_m_test))
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
            # HTR's nuisance heads are its own evidence; they also join the
            # shared ensemble, which the other architectures do not require.
            self.timer.lap("htr_nuisance_fits", attribution=HTR_NEURAL)

        if not nuisance_train:
            raise ValueError("multi_model_forest Stage 1 requires at least one nuisance source")
        e_train = np.nanmean(np.vstack([item[1] for item in nuisance_train]), axis=0)
        m_train = np.nanmean(np.vstack([item[2] for item in nuisance_train]), axis=0)
        e_test = np.nanmean(np.vstack([item[1] for item in nuisance_test]), axis=0)
        m_test = np.nanmean(np.vstack([item[2] for item in nuisance_test]), axis=0)
        binary_dina = str(self.config.outcome_type).lower() == "binary"
        dina_train = None
        if binary_dina:
            from .stage1_dina import crossfit_nuisance

            dina_train, _ = crossfit_nuisance(
                texts_train, t, y, folds=self.nn_config.nuisance_folds, seed=61_000 + outer_fold
            )
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
        if binary_dina:
            for key, value in dina_train.items():
                ensemble_nuisance_train["dina_" + key] = value
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
        self.timer.lap(
            "nuisance_ensemble",
            attribution=SHARED_ATTRIBUTION,
            consumers=TEXT_MODEL_ARCHITECTURES,
        )

        matched_pair_lane: Optional[MatchedPairLaneResult] = None
        if self._matched_pair_uplift_enabled():
            matched_pair_lane = self._fit_matched_pair_lane(
                train_df=train_df,
                test_df=test_df,
                texts_train=texts_train,
                texts_test=texts_test,
                y=y,
                t=t,
                bow_nuisance_by_view=bow_nuisance_by_view,
                dina_train=dina_train,
                outer_fold=outer_fold,
            )
            prediction_frames.extend(matched_pair_lane.prediction_frames)
            metrics.update(matched_pair_lane.metrics)

        if self._matched_pair_uplift_enabled():
            self.timer.lap("matched_pair_uplift_fits", attribution=MATCHED_PAIR_UPLIFT)

        if self._bow_enabled():
            for view_index, view in enumerate(self.nn_config.bow_views):
                pseudo_unit = f"bow/effect/view_{int(view_index):03d}_pseudo_outcome"
                pseudo_train, pseudo_test = self._checkpointed_value(
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
                        dina_data=(
                            (t, y) if str(self.config.outcome_type).lower() == "binary" else None
                        ),
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
                weighted_unit = f"bow/effect/view_{int(view_index):03d}_weighted_r"
                r_train, r_test = self._checkpointed_value(
                    weighted_unit,
                    lambda: (
                        (pseudo_train, pseudo_test)
                        if binary_dina
                        else self._fit_bow_regression_train_test(
                            texts_train,
                            texts_test,
                            pseudo_target,
                            r_weight,
                            outer_fold=outer_fold,
                            view=view,
                            view_index=view_index,
                            target_name="effect_weighted_r",
                            dina_data=(
                                (t, y)
                                if str(self.config.outcome_type).lower() == "binary"
                                else None
                            ),
                            seed_offset=70_000,
                            train_row_ids=train_df["_oci_row_id"].to_numpy(),
                            test_row_ids=test_df["_oci_row_id"].to_numpy(),
                            checkpoint_namespace=weighted_unit,
                            checkpoint_dependencies=("assembly/nuisance_ensemble",),
                        )
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
                self.timer.lap("bow_effect_fits", attribution=BOW_R_LOSS)
                prediction_frames.append(
                    _source_prediction_frame(
                        train_df,
                        test_df,
                        outer_fold=outer_fold,
                        source_name=f"bow__{view.name}__effect",
                        values=(
                            {"tau_hat_dina": (pseudo_train, pseudo_test)}
                            if binary_dina
                            else {
                                "tau_hat_pseudo_target": (pseudo_train, pseudo_test),
                                "tau_hat_weighted_r": (r_train, r_test),
                            }
                        ),
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
                    self.timer.lap("bow_importance_effect_terms", attribution=BOW_R_LOSS)

        if self._htr_enabled():
            htr_effect_variants: Dict[str, Any] = {}
            htr_provider = self._htr_provider()
            variants = (
                [("dina", "effect_dina_tau_pred")]
                if str(self.config.outcome_type).lower() == "binary"
                else [
                    ("pseudo_outcome_mse", "effect_pseudo_target_pred"),
                    ("squared_r_loss", "effect_weighted_r_tau_pred"),
                ]
            )
            for effect_objective, feature_suffix in variants:
                htr_effect_bundle = htr_provider.fit_effect_variant_inner_ensemble_predict(
                    train_df,
                    test_df,
                    ensemble_nuisance_train,
                    outer_fold,
                    effect_objective=effect_objective,
                )
                htr_effect_train = htr_effect_bundle["train"]
                test_predictions = htr_effect_bundle["test_predictions"]
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
            self.timer.lap("htr_effect_fits", attribution=HTR_NEURAL)

        importance: Dict[str, Any] = _multi_view_importance(
            bow_view_results,
            top_n=int(self.nn_config.top_n_features),
        )
        importance["feature_discovery_methods"] = self._enabled_feature_discovery_methods()
        if matched_pair_lane is not None:
            importance["matched_pair_uplift"] = matched_pair_lane.importance
        if bow_view_results:
            # Metadata only: the per-view effect evidence lives once, under
            # importance["views"]; a copy here would be emitted twice downstream.
            nuisance_source_names = [item[0] for item in nuisance_train]
            if binary_dina:
                importance["ensemble_r"] = {
                    "target_source": "per_view_cross_fitted_dina_nuisance",
                    "nuisance_sources": nuisance_source_names,
                    "pseudo_target_construction": (
                        "binary outcome: Bernoulli DINA effect model on each "
                        "view's BoW features, offset by that view's "
                        "cross-fitted DINA nuisances (treatment propensity and "
                        "arm-specific outcome models); the ensemble mean "
                        "nuisance does not enter the effect target"
                    ),
                }
            else:
                importance["ensemble_r"] = {
                    "target_source": (
                        "ensemble_mean_nuisance_with_htr"
                        if any(str(name).startswith("htr") for name in nuisance_source_names)
                        else "ensemble_mean_nuisance"
                    ),
                    "nuisance_sources": nuisance_source_names,
                    "pseudo_target_construction": (
                        "mean nuisance predictions across Stage 1 text models, then "
                        "(Y - mean_m_hat) / (T - mean_e_hat), regressed with "
                        "weights (T - mean_e_hat)^2"
                    ),
                }

        self.timer.lap(
            "evidence_assembly",
            attribution=SHARED_ATTRIBUTION,
            consumers=TEXT_MODEL_ARCHITECTURES,
        )
        embedding_evidence: Dict[str, Any] = {}
        if self._embedding_contrast_enabled():
            embedding_evidence, direction_frames = (
                self._build_primary_embedding_contrast_evidence(
                    discovery_df=train_df,
                    heldout_df=test_df,
                    outer_fold=outer_fold,
                    y=y,
                    t=t,
                    pseudo_target=pseudo_target,
                    t_resid=t_resid,
                    importance=importance,
                )
            )
            prediction_frames.extend(direction_frames)

        handoff_evidence = {
            "metrics": copy.deepcopy(metrics),
            "importance": importance,
            "embedding_contrast_evidence": embedding_evidence,
            "htr_evidence": htr_evidence,
        }

        return _FeatureBundle(
            prediction_frames=prediction_frames,
            metrics=metrics,
            handoff_evidence=handoff_evidence,
        )

    def _fit_matched_pair_lane(
        self,
        *,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        texts_train: Sequence[str],
        texts_test: Sequence[str],
        y: np.ndarray,
        t: np.ndarray,
        bow_nuisance_by_view: Sequence[Mapping[str, Any]],
        dina_train: Optional[Mapping[str, np.ndarray]],
        outer_fold: int,
    ) -> MatchedPairLaneResult:
        """Match and fit ``matched_pair_uplift`` on this context's training rows.

        Both matching scores are out of fold for every training row:

        * the propensity is the mean of the per-view BoW nuisance
          propensities, each predicted by a model fitted on the other folds
          (the HTR propensity is excluded because its out-of-fold calibrator
          is fitted on predictions from models that saw the row);
        * the prognostic score is, for binary outcomes, the untreated-arm
          probability of the context's DINA nuisance cross-fit and, for
          continuous outcomes, a cross-fitted elastic net on untreated rows.
        """

        if not bow_nuisance_by_view:
            raise RuntimeError("matched_pair_uplift requires the BoW nuisance fits")
        settings = MatchedPairSettings.from_config(
            self.nn_config,
            outcome_type=str(self.config.outcome_type),
            top_n=int(self.nn_config.top_n_features),
        )
        propensity = np.nanmean(
            np.vstack([np.asarray(item["e_hat"], dtype=float) for item in bow_nuisance_by_view]),
            axis=0,
        )
        dependencies: Tuple[str, ...] = tuple(
            f"bow/nuisance/view_{int(item['view_index']):03d}_treatment"
            for item in bow_nuisance_by_view
        )
        groups = distinct_vectorizer_groups(self.nn_config.bow_views)
        if settings.binary:
            if dina_train is None:
                raise RuntimeError("binary matched_pair_uplift requires the DINA nuisance cross-fit")
            prognostic = np.asarray(dina_train["mu0"], dtype=float)
            prognostic_audit: Dict[str, Any] = {
                "source": BINARY_PROGNOSTIC_SOURCE,
                "model": (
                    "ridge logistic regression on untreated fitting rows "
                    "(dina_auxiliary BoW view), stratified K-fold cross-fit"
                ),
                "folds": int(self.nn_config.nuisance_folds),
                "seed": int(61_000 + outer_fold),
            }
        else:
            unit = "bow/matched_pair/prognostic_score"
            first_group = groups[0]
            seed = matched_pair_seed(self.config.seed, outer_fold) + 1

            def compute_prognostic() -> Dict[str, Any]:
                return crossfit_prognostic_score(
                    texts_train,
                    y,
                    t,
                    vectorizer_params=first_group["vectorizer_params"],
                    folds=int(self.nn_config.nuisance_folds),
                    l1_ratio=settings.l1_ratio,
                    cv_folds=settings.cv_folds,
                    max_iter=settings.max_iter,
                    seed=seed,
                )

            def validate_prognostic(payload: Mapping[str, Any]) -> None:
                values = np.asarray(payload.get("prognostic"), dtype=float)
                if values.shape != (len(train_df),) or not np.isfinite(values).all():
                    raise ValueError("matched-pair prognostic checkpoint is invalid")

            payload = self._checkpointed_value(
                unit,
                compute_prognostic,
                train_df=train_df,
                test_df=test_df,
                parameters={
                    "checkpoint_version": MATCHED_PAIR_CHECKPOINT_VERSION,
                    "outer_fold": int(outer_fold),
                    "vectorizer": first_group["vectorizer_params"],
                    "folds": int(self.nn_config.nuisance_folds),
                    "settings": settings.identity(),
                    "seed": int(seed),
                },
                validator=validate_prognostic,
            )
            prognostic = np.asarray(payload["prognostic"], dtype=float)
            prognostic_audit = {
                **dict(payload["audit"]),
                "vectorizer_view": str(first_group["name"]),
            }
            dependencies = (*dependencies, unit)
        score_sources = {
            "propensity": {
                "source": PROPENSITY_SOURCE,
                "views": [str(item["view"].name) for item in bow_nuisance_by_view],
                "out_of_fold": True,
            },
            "prognostic": {**prognostic_audit, "out_of_fold": True},
        }

        def checkpoint(
            unit: str,
            compute: Callable[[], Any],
            *,
            parameters: Mapping[str, Any],
            validator: Callable[[Any], None],
        ) -> Any:
            return self._checkpointed_value(
                unit,
                compute,
                train_df=train_df,
                test_df=test_df,
                parameters={**dict(parameters), "score_sources": score_sources},
                dependencies=dependencies,
                validator=validator,
            )

        return run_matched_pair_lane(
            train_df=train_df,
            test_df=test_df,
            texts_train=texts_train,
            texts_test=texts_test,
            y_train=y,
            t_train=t,
            propensity=propensity,
            prognostic=prognostic,
            score_sources=score_sources,
            vectorizer_groups=groups,
            settings=settings,
            outer_fold=int(outer_fold),
            seed=matched_pair_seed(self.config.seed, outer_fold),
            checkpoint=checkpoint,
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
        vectorizer = _make_bow_vectorizer(_vectorizer_params(view))
        x_model = vectorizer.fit_transform(texts)
        features = np.asarray(vectorizer.get_feature_names_out())

        # Constant zero scores (single-class targets) are treated as signed;
        # tree ensembles report unsigned importances and get no negative list.
        signed = {"treatment": True, "outcome": True, "pseudo_target": True}
        if len(np.unique(np.asarray(t, dtype=int))) < 2:
            treatment_coef = np.zeros(len(features), dtype=float)
        else:
            treatment_model = _make_bow_classifier(_model_params(view), random_state=101)
            treatment_model.fit(x_model, np.asarray(t, dtype=int))
            treatment_coef = _model_feature_scores(treatment_model, len(features))
            signed["treatment"] = _model_feature_scores_are_signed(treatment_model)

        if str(self.config.outcome_type).lower() == "continuous":
            outcome_model = _make_bow_regressor(_model_params(view), random_state=202)
            outcome_model.fit(x_model, y)
            outcome_coef = _model_feature_scores(outcome_model, len(features))
            signed["outcome"] = _model_feature_scores_are_signed(outcome_model)
        elif len(np.unique(np.asarray(y, dtype=int))) < 2:
            outcome_coef = np.zeros(len(features), dtype=float)
        else:
            outcome_model = _make_bow_classifier(_model_params(view), random_state=202)
            outcome_model.fit(x_model, np.asarray(y, dtype=int))
            outcome_coef = _model_feature_scores(outcome_model, len(features))
            signed["outcome"] = _model_feature_scores_are_signed(outcome_model)
        self.timer.lap("bow_importance_nuisance_terms", attribution=BOW_NUISANCE)

        if str(self.config.outcome_type).lower() == "binary":
            from .stage1_dina import crossfit_nuisance
            from ..models import dina

            dn, _ = crossfit_nuisance(
                texts, t, y, view=view, folds=self.nn_config.nuisance_folds, seed=303
            )
            effect_model = dina.fit(
                x_model, y, t, dn["a"], dn["nu"], regularization=view.ridge_alpha / len(y)
            )
        else:
            effect_model = _make_bow_regressor(_model_params(view), random_state=303)
            _fit_regressor(
                effect_model,
                x_model,
                pseudo_target,
                sample_weight=pseudo_target_sample_weight,
                unsupported_sample_weight_policy=(view.unsupported_sample_weight_policy),
            )
        effect_coef = _model_feature_scores(effect_model, len(features))
        signed["pseudo_target"] = _model_feature_scores_are_signed(effect_model)
        self.timer.lap("bow_importance_effect_terms", attribution=BOW_R_LOSS)

        top_n = int(self.nn_config.top_n_features)
        confounder_score = np.abs(treatment_coef) * np.abs(outcome_coef)
        ranked: Dict[str, Any] = {}
        for target, coef in (
            ("treatment", treatment_coef),
            ("outcome", outcome_coef),
            ("pseudo_target", effect_coef),
        ):
            ranked[f"{target}_positive"] = _top_feature_rows(features, coef, top_n)
            if signed[target]:
                ranked[f"{target}_negative"] = _top_feature_rows(
                    features,
                    coef,
                    top_n,
                    descending=False,
                )
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
            "score_polarity": {
                target: ("signed" if is_signed else "unsigned_importance")
                for target, is_signed in signed.items()
            },
            **ranked,
        }

    def _build_primary_embedding_contrast_evidence(
        self,
        *,
        discovery_df: pd.DataFrame,
        heldout_df: pd.DataFrame,
        outer_fold: int,
        y: np.ndarray,
        t: np.ndarray,
        pseudo_target: np.ndarray,
        t_resid: np.ndarray,
        importance: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[pd.DataFrame]]:
        """Return the handed-off embedding evidence and its held-out row scores.

        The row scores project the context's held-out rows on exactly the
        directions whose aligned chunks enter the handoff.  They exist only for
        post-hoc evaluation; no fit, ranking, or selection reads them.
        """

        if not self._embedding_contrast_enabled():
            return {}, []
        generator = self._embedding_generator()
        generator.prepare(self.dataset)
        self.timer.lap(
            "embedding_cache_open",
            attribution=SHARED_ATTRIBUTION,
            consumers=EMBEDDING_ARCHITECTURES,
        )
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
        dina_nuisance = None
        if str(self.config.outcome_type).lower() == "binary":
            from .stage1_dina import crossfit_nuisance

            dina_nuisance, _ = crossfit_nuisance(
                discovery_df[self.config.text_column].astype(str).tolist(),
                t,
                y,
                folds=self.nn_config.nuisance_folds,
                seed=49_001,
            )
            self.timer.lap(
                "embedding_dina_nuisance",
                attribution=SHARED_ATTRIBUTION,
                consumers=EMBEDDING_ARCHITECTURES,
            )
        evidence, directions = generator.build_evidence_with_directions(
            dina_nuisance=dina_nuisance,
            discovery_df=discovery_df,
            y=y,
            t=t,
            pseudo_target=[pseudo_target],
            t_resid=[t_resid],
            pseudo_target_names=["stage1_ensemble_mean_nuisance"],
            importance=importance,
            timer=self.timer,
        )
        frames = _handed_off_direction_frames(
            generator,
            heldout_df,
            directions,
            outer_fold=outer_fold,
        )
        self.timer.lap(
            "embedding_heldout_projection",
            attribution=SHARED_ATTRIBUTION,
            consumers=(EMBEDDING_WHOLE_COHORT, EMBEDDING_CLUSTERED),
        )
        return evidence, frames

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
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        n_jobs = self._fold_n_jobs(len(split_items))
        logger.info(
            "Outer fold %s BoW binary %s view=%s model=%s folds=%s n_jobs=%s " "backend=%s",
            outer_fold,
            label_name,
            view.name,
            view.bow_model,
            len(split_items),
            n_jobs,
            "threading",
        )

        def compute_fold(fold: int, fit_pos: np.ndarray, heldout_pos: np.ndarray):
            fit_pos = np.asarray(fit_pos, dtype=int)
            heldout_pos = np.asarray(heldout_pos, dtype=int)
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
            return {
                "fold": int(fold),
                "heldout_pos": heldout_pos,
                "heldout_pred": heldout_pred,
                "test_pred": test_pred,
            }

        def run_fold(fold: int, fit_pos: np.ndarray, heldout_pos: np.ndarray):
            store = self.checkpoint_store
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
        for result in fold_results:
            heldout_pos = result["heldout_pos"]
            oof[heldout_pos] = result["heldout_pred"]
            test_predictions.append(np.asarray(result["test_pred"], dtype=float))
        test_pred = np.nanmean(np.vstack(test_predictions), axis=0)
        return (
            np.clip(oof, self.nn_config.e_clip, 1.0 - self.nn_config.e_clip),
            np.clip(test_pred, self.nn_config.e_clip, 1.0 - self.nn_config.e_clip),
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
        dina_data=None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if dina_data is not None:
            from .stage1_dina import crossfit_effect

            t, y = dina_data
            fitted = crossfit_effect(
                texts_train,
                t,
                y,
                test_texts=texts_test,
                view=view,
                folds=self.nn_config.effect_folds,
                seed=13_000 + int(seed_offset) + 100 * outer_fold + 1000 * view_index,
            )
            return fitted["tau"], fitted["test_tau"]
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
        n_jobs = self._fold_n_jobs(len(split_items))
        logger.info(
            "Outer fold %s BoW regression %s view=%s model=%s folds=%s n_jobs=%s " "backend=%s",
            outer_fold,
            target_name,
            view.name,
            view.bow_model,
            len(split_items),
            n_jobs,
            "threading",
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
            return {
                "fold": int(fold),
                "heldout_pos": heldout_pos,
                "heldout_pred": heldout_pred,
                "test_pred": test_pred,
            }

        def run_fold(fold: int, fit_pos: np.ndarray, heldout_pos: np.ndarray):
            store = self.checkpoint_store
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
        for result in fold_results:
            heldout_pos = result["heldout_pos"]
            oof[heldout_pos] = result["heldout_pred"]
            test_predictions.append(np.asarray(result["test_pred"], dtype=float))
        test_pred = np.nanmean(np.vstack(test_predictions), axis=0)
        return oof, test_pred

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
        # A bag-of-words lane for both outcome types: it needs the BoW views
        # and their out-of-fold propensities.
        return self._bow_enabled() and bool(
            getattr(self.nn_config, "matched_pair_uplift_enabled", True)
        )

    def _embedding_generator(self) -> EmbeddingContrastEvidenceGenerator:
        if self.embedding_evidence_generator is None:
            self.embedding_evidence_generator = EmbeddingContrastEvidenceGenerator(
                config=self.config,
                output_dir=self.artifact_dir,
            )
        return self.embedding_evidence_generator

    def _htr_provider(self) -> MultiModelForestStage1HTRProvider:
        if self._default_htr_provider is None:
            self._default_htr_provider = MultiModelForestStage1HTRProvider(
                config=self.config,
                device=self.device,
                checkpoint_store=self.checkpoint_store,
                num_workers=self.num_workers,
            )
        return self._default_htr_provider

    def _fold_n_jobs(self, folds: int) -> int:
        if folds <= 0:
            return 1
        return max(1, min(int(self.num_workers), int(folds)))

    def _run_fold_tasks(self, run_fold: Any, split_items: Sequence[Any]) -> List[Any]:
        # Independent BoW folds share the context's data through threads.
        n_jobs = self._fold_n_jobs(len(split_items))
        if n_jobs <= 1:
            return [
                run_fold(int(fold), np.asarray(fit_pos), np.asarray(heldout_pos))
                for fold, (fit_pos, heldout_pos) in split_items
            ]
        return Parallel(
            n_jobs=n_jobs,
            backend="threading",
            batch_size=1,
            pre_dispatch="all",
        )(
            delayed(run_fold)(int(fold), np.asarray(fit_pos), np.asarray(heldout_pos))
            for fold, (fit_pos, heldout_pos) in split_items
        )


def _vectorizer_params(view: BoWViewConfig) -> Dict[str, Any]:
    return _bow_vectorizer_params(view)


def _model_params(view: BoWViewConfig) -> Dict[str, Any]:
    return _bow_model_params(view)


def _bow_view_to_dict(view: BoWViewConfig) -> Dict[str, Any]:
    return asdict(view)


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


def _handed_off_direction_frames(
    generator: EmbeddingContrastEvidenceGenerator,
    heldout_df: pd.DataFrame,
    directions: Sequence[Mapping[str, Any]],
    *,
    outer_fold: int,
) -> List[pd.DataFrame]:
    """Score held-out rows on the exact handed-off embedding directions.

    One wide frame per embedding architecture, one column per direction and
    statistic (mean and maximum chunk cosine).  Only held-out rows are
    written: the directions were fitted on this context's training rows, so
    their training-row projections would be in-sample.
    """

    row_ids = heldout_df["_oci_row_id"].astype(int).tolist()
    frames: List[pd.DataFrame] = []
    for architecture, source_name in EMBEDDING_DIRECTION_SCORE_SOURCES.items():
        selected = [item for item in directions if item["architecture"] == architecture]
        if not selected or not row_ids:
            continue
        means, maxes = generator.chunk_cosine_summaries(
            row_ids,
            [item["direction"] for item in selected],
        )
        payload: Dict[str, Any] = {
            "_oci_row_id": np.asarray(row_ids, dtype=int),
            "outer_fold": int(outer_fold),
            "split_role": "test_outer_train_fit",
            "source_name": source_name,
        }
        for index, item in enumerate(selected):
            for statistic, values in zip(
                EMBEDDING_DIRECTION_SCORE_STATISTICS,
                (means, maxes),
            ):
                column = embedding_direction_score_column(item["name"], statistic)
                if column in payload:
                    raise ValueError(
                        f"two handed-off embedding directions share column {column!r}"
                    )
                payload[column] = np.asarray(values[:, index], dtype=float)
        frames.append(pd.DataFrame(payload))
    return frames


def _source_prediction_architecture(source_name: Any) -> str:
    """Map a row-score producer onto the public Stage 1 architecture contract."""

    source = str(source_name).lower()
    for architecture, handed_off_source in EMBEDDING_DIRECTION_SCORE_SOURCES.items():
        if source == handed_off_source:
            return architecture
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


def _finite_or_none(value: Any) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value
