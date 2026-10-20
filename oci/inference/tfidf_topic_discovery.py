"""Honest TF-IDF nuisance modeling and three-bank topic discovery.

This module is intentionally independent of the HTR, embedding, and uplift
implementations.  One exact context fits its vectorizers, nuisance stacks, and
topic models on fit rows only, then predicts its external held-out rows
without refitting.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from threadpoolctl import threadpool_limits

from ..config import (
    BoWViewConfig,
    LogisticRegressionScientificConfig,
    NuisanceCalibrationScientificConfig,
    RidgeScientificConfig,
    TfidfNuisanceStackScientificConfig,
    TfidfTopicDiscoveryConfig,
)
from .sparse_text_modeling import (
    _bow_model_params,
    _bow_vectorizer_params,
    _make_bow_classifier,
    _make_bow_regressor,
    _make_bow_vectorizer,
)
from .sklearn_nmf_compat import fit_transform_nmf_with_c_order_initial_w
from .tfidf_topic_score_selection import (
    ORPHAN_NGRAM_SCORE_SCHEMA_VERSION,
    score_effect_orphan_ngram_clusters,
)

logger = logging.getLogger(__name__)

HANDOFF_SCHEMA_VERSION = "multi_model_forest_handoff_v2"
DISCOVERY_SCHEMA_VERSION = "tfidf_topic_discovery_v4_dina"


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def row_set_fingerprint(row_ids: Sequence[Any]) -> str:
    """Fingerprint a row set without making its input order meaningful."""
    values = sorted(str(value) for value in row_ids)
    return stable_hash(values)


def _normalize_texts(values: Sequence[Any]) -> List[str]:
    # Case/token handling belongs exclusively to the authenticated vectorizer
    # contract. Consuming the complete supplied string here avoids a second,
    # hidden normalization path that differs from role-neutral BoW execution.
    return [str(value or "") for value in values]


def _bounded_folds(requested: int, labels: np.ndarray, *, stratified: bool) -> int:
    n_rows = int(len(labels))
    if n_rows < 4:
        raise ValueError("At least four rows are required for nested nuisance cross-fitting")
    upper = n_rows
    if stratified:
        counts = np.unique(labels.astype(int), return_counts=True)[1]
        if len(counts) < 2:
            return 2
        upper = int(np.min(counts))
    return max(2, min(int(requested), upper, n_rows // 2))


def _splitter(
    labels: np.ndarray,
    folds: int,
    *,
    stratified: bool,
    seed: int,
    shuffle: bool,
):
    counts = np.unique(labels.astype(int), return_counts=True)[1] if stratified else []
    random_state = int(seed) if bool(shuffle) else None
    if stratified and len(counts) >= 2 and int(np.min(counts)) >= int(folds):
        return StratifiedKFold(
            n_splits=folds,
            shuffle=bool(shuffle),
            random_state=random_state,
        ).split(
            np.zeros(len(labels)), labels.astype(int)
        )
    return KFold(
        n_splits=folds,
        shuffle=bool(shuffle),
        random_state=random_state,
    ).split(np.zeros(len(labels)))


def _vectorizer(view: BoWViewConfig) -> TfidfVectorizer:
    return _make_bow_vectorizer(_bow_vectorizer_params(view))


def _classifier(view: BoWViewConfig, seed: int):
    return _make_bow_classifier(
        _bow_model_params(view),
        random_state=int(seed),
    )


def _regressor(view: BoWViewConfig, seed: int):
    return _make_bow_regressor(
        _bow_model_params(view),
        random_state=int(seed),
    )


def _constant_prediction(values: np.ndarray, n_rows: int) -> np.ndarray:
    mean = float(np.mean(values)) if len(values) else 0.0
    return np.full(int(n_rows), mean, dtype=float)


@dataclass
class CrossFittedStack:
    """Fitted complete-context bases and a stack trained on context OOF scores."""

    views: List[BoWViewConfig]
    binary: bool
    base_models: List[Tuple[Optional[TfidfVectorizer], Optional[Any], float]]
    stack_model: Optional[Any]
    stack_constant: float
    config_hash: str

    def predict(self, texts: Sequence[str]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        texts = _normalize_texts(texts)
        columns: List[np.ndarray] = []
        by_view: Dict[str, np.ndarray] = {}
        for view, (vectorizer, model, constant) in zip(self.views, self.base_models):
            if vectorizer is None or model is None:
                prediction = np.full(len(texts), constant, dtype=float)
            else:
                matrix = vectorizer.transform(texts)
                prediction = (
                    model.predict_proba(matrix)[:, 1] if self.binary else model.predict(matrix)
                )
            prediction = np.asarray(prediction, dtype=float)
            columns.append(prediction)
            by_view[view.name] = prediction
        meta = np.column_stack(columns)
        if self.stack_model is None:
            stacked = np.full(len(texts), self.stack_constant, dtype=float)
        elif self.binary:
            stacked = self.stack_model.predict_proba(meta)[:, 1]
        else:
            stacked = self.stack_model.predict(meta)
        return np.asarray(stacked, dtype=float), by_view


def _configured_logistic_regression(
    *,
    c: float,
    max_iter: int,
    config: LogisticRegressionScientificConfig,
    seed: int,
) -> LogisticRegression:
    return LogisticRegression(
        penalty=config.penalty,
        dual=bool(config.dual),
        tol=float(config.tol),
        C=float(c),
        fit_intercept=bool(config.fit_intercept),
        intercept_scaling=float(config.intercept_scaling),
        class_weight=config.class_weight,
        random_state=int(seed),
        solver=str(config.solver),
        max_iter=int(max_iter),
        verbose=0,
        warm_start=bool(config.warm_start),
        n_jobs=1,
        l1_ratio=config.l1_ratio,
    )


def _configured_ridge(
    *,
    alpha: float,
    config: RidgeScientificConfig,
    seed: int,
) -> Ridge:
    return Ridge(
        alpha=float(alpha),
        fit_intercept=bool(config.fit_intercept),
        copy_X=True,
        max_iter=config.max_iter,
        tol=float(config.tol),
        solver=str(config.solver),
        positive=bool(config.positive),
        random_state=int(seed),
    )


def _fit_stack(
    meta: np.ndarray,
    values: np.ndarray,
    *,
    binary: bool,
    seed: int,
    config: TfidfNuisanceStackScientificConfig,
):
    constant = float(np.mean(values))
    if meta.shape[1] == 0 or (binary and len(np.unique(values.astype(int))) < 2):
        return None, constant
    if binary:
        model = _configured_logistic_regression(
            c=config.meta_logistic_c,
            max_iter=config.meta_logistic_max_iter,
            config=config.meta_logistic,
            seed=seed,
        )
        model.fit(meta, values.astype(int))
    else:
        model = _configured_ridge(
            alpha=config.meta_ridge_alpha,
            config=config.meta_ridge,
            seed=seed,
        )
        model.fit(meta, values)
    return model, constant


def _vectorizer_key(view: BoWViewConfig) -> Tuple[Any, ...]:
    return (stable_hash(_bow_vectorizer_params(view)),)


def _matrix_model_prediction(
    x_fit: Any,
    x_predict: Any,
    values: np.ndarray,
    view: BoWViewConfig,
    *,
    binary: bool,
    seed: int,
) -> Tuple[np.ndarray, Optional[Any]]:
    if binary and len(np.unique(values.astype(int))) < 2:
        return _constant_prediction(values, x_predict.shape[0]), None
    model = _classifier(view, seed) if binary else _regressor(view, seed)
    model.fit(x_fit, values.astype(int) if binary else values)
    if x_predict.shape[0] == 0:
        return np.zeros(0, dtype=float), model
    prediction = model.predict_proba(x_predict)[:, 1] if binary else model.predict(x_predict)
    return np.asarray(prediction, dtype=float), model


@dataclass(frozen=True)
class _JointNuisanceTopLevelFoldTask:
    """Complete immutable input for one top-level nuisance fold."""

    fold: int
    fit_positions: np.ndarray
    heldout_positions: np.ndarray
    texts: Tuple[str, ...]
    treatment: np.ndarray
    outcome: np.ndarray
    outcome_binary: bool
    split_labels: np.ndarray
    views: Tuple[BoWViewConfig, ...]
    fold_count: int
    random_state: int
    nuisance_stack_config: TfidfNuisanceStackScientificConfig


@dataclass(frozen=True)
class _JointNuisanceTopLevelFoldResult:
    """Isolated CPU result returned by one top-level nuisance worker."""

    fold: int
    fit_positions: np.ndarray
    heldout_positions: np.ndarray
    base_predictions: Mapping[str, np.ndarray]
    stacked_predictions: Mapping[str, np.ndarray]


def _fit_joint_nuisance_top_level_fold(
    task: _JointNuisanceTopLevelFoldTask,
) -> _JointNuisanceTopLevelFoldResult:
    """Fit one outer nuisance fold; its subfold loop is deliberately serial."""

    if not isinstance(task, _JointNuisanceTopLevelFoldTask):
        raise TypeError("joint nuisance worker requires its typed fold task")
    fit_pos = np.asarray(task.fit_positions, dtype=int)
    heldout_pos = np.asarray(task.heldout_positions, dtype=int)
    texts = task.texts
    targets = {
        "treatment": (np.asarray(task.treatment, dtype=float), True),
        "outcome": (
            np.asarray(task.outcome, dtype=float),
            bool(task.outcome_binary),
        ),
    }
    if task.outcome_binary:
        targets.update(
            mu0=(np.asarray(task.outcome, float), True), mu1=(np.asarray(task.outcome, float), True)
        )

    def target_rows(target, positions):
        return (
            np.asarray(task.treatment)[positions] == int(target[-1])
            if target in {"mu0", "mu1"}
            else np.ones(len(positions), bool)
        )

    split_labels = np.asarray(task.split_labels, dtype=int)
    views = list(task.views)
    grouped: Dict[Tuple[Any, ...], List[Tuple[int, BoWViewConfig]]] = {}
    for index, view in enumerate(views):
        grouped.setdefault(_vectorizer_key(view), []).append((index, view))

    # This is the only numeric-library limiter inside a worker. The subfold
    # iterator below remains an ordinary serial loop and never constructs a
    # nested Joblib pool.
    with threadpool_limits(limits=1):
        fit_strata = split_labels[fit_pos]
        sub_folds = _bounded_folds(
            min(task.fold_count, len(fit_pos) // 2),
            fit_strata,
            stratified=True,
        )
        meta_fit = {
            target: np.full(
                (len(fit_pos), len(views)),
                np.nan,
                dtype=float,
            )
            for target in targets
        }
        for sub_fold, (sub_fit_local, sub_hold_local) in enumerate(
            _splitter(
                fit_strata,
                sub_folds,
                stratified=True,
                seed=task.random_state + 1000 * task.fold,
                shuffle=task.nuisance_stack_config.split_shuffle,
            ),
            start=1,
        ):
            sub_fit_local = np.asarray(sub_fit_local, dtype=int)
            sub_hold_local = np.asarray(sub_hold_local, dtype=int)
            sub_fit_pos = fit_pos[sub_fit_local]
            sub_hold_pos = fit_pos[sub_hold_local]
            for group_index, group_views in enumerate(grouped.values()):
                vectorizer = _vectorizer(group_views[0][1])
                try:
                    x_sub_fit = vectorizer.fit_transform(
                        [texts[index] for index in sub_fit_pos]
                    )
                    x_sub_hold = vectorizer.transform(
                        [texts[index] for index in sub_hold_pos]
                    )
                except ValueError as exc:
                    if group_views[0][1].empty_vocabulary_policy == "fail_closed":
                        raise ValueError(
                            "joint nuisance subfold could not fit the configured "
                            f"vocabulary for view {group_views[0][1].name!r}"
                        ) from exc
                    x_sub_fit = x_sub_hold = None
                for view_index, view in group_views:
                    for target_index, (target, (values, binary)) in enumerate(
                        targets.items()
                    ):
                        mask = target_rows(target, sub_fit_pos)
                        if not mask.any():
                            raise ValueError("DINA nuisance subfold lacks a treatment arm")
                        if x_sub_fit is None:
                            prediction = _constant_prediction(
                                values[sub_fit_pos][mask],
                                len(sub_hold_pos),
                            )
                        else:
                            prediction, _ = _matrix_model_prediction(
                                x_sub_fit[mask],
                                x_sub_hold,
                                values[sub_fit_pos][mask],
                                view,
                                binary=binary,
                                seed=(
                                    task.random_state
                                    + 10_000 * task.fold
                                    + 1000 * sub_fold
                                    + 100 * group_index
                                    + 10 * view_index
                                    + target_index
                                ),
                            )
                        meta_fit[target][sub_hold_local, view_index] = prediction

        base_predictions = {
            target: np.full(
                (len(heldout_pos), len(views)),
                np.nan,
                dtype=float,
            )
            for target in targets
        }
        for group_index, group_views in enumerate(grouped.values()):
            vectorizer = _vectorizer(group_views[0][1])
            try:
                x_fit = vectorizer.fit_transform(
                    [texts[index] for index in fit_pos]
                )
                x_heldout = vectorizer.transform(
                    [texts[index] for index in heldout_pos]
                )
            except ValueError as exc:
                if group_views[0][1].empty_vocabulary_policy == "fail_closed":
                    raise ValueError(
                        "joint nuisance fold could not fit the configured "
                        f"vocabulary for view {group_views[0][1].name!r}"
                    ) from exc
                x_fit = x_heldout = None
            for view_index, view in group_views:
                for target_index, (target, (values, binary)) in enumerate(
                    targets.items()
                ):
                    mask = target_rows(target, fit_pos)
                    if not mask.any():
                        raise ValueError("DINA nuisance fold lacks a treatment arm")
                    if x_fit is None:
                        prediction = _constant_prediction(
                            values[fit_pos][mask],
                            len(heldout_pos),
                        )
                    else:
                        prediction, _ = _matrix_model_prediction(
                            x_fit[mask],
                            x_heldout,
                            values[fit_pos][mask],
                            view,
                            binary=binary,
                            seed=(
                                task.random_state
                                + 30_000 * task.fold
                                + 100 * group_index
                                + 10 * view_index
                                + target_index
                            ),
                        )
                    base_predictions[target][:, view_index] = prediction

        stacked_predictions: Dict[str, np.ndarray] = {}
        for target_index, (target, (values, binary)) in enumerate(targets.items()):
            if not np.isfinite(meta_fit[target]).all():
                raise RuntimeError(
                    f"Nested {target} nuisance meta-features are incomplete"
                )
            mask = target_rows(target, fit_pos)
            stack_model, stack_constant = _fit_stack(
                meta_fit[target][mask],
                values[fit_pos][mask],
                binary=binary,
                seed=(task.random_state + 40_000 + 10 * task.fold + target_index),
                config=task.nuisance_stack_config,
            )
            held_meta = base_predictions[target]
            if stack_model is None:
                stacked = np.full(
                    len(heldout_pos),
                    stack_constant,
                    dtype=float,
                )
            elif binary:
                stacked = stack_model.predict_proba(held_meta)[:, 1]
            else:
                stacked = stack_model.predict(held_meta)
            stacked_predictions[target] = np.asarray(stacked, dtype=float)

    return _JointNuisanceTopLevelFoldResult(
        fold=int(task.fold),
        fit_positions=fit_pos,
        heldout_positions=heldout_pos,
        base_predictions=base_predictions,
        stacked_predictions=stacked_predictions,
    )


def _joint_nuisance_workers(
    *,
    tfidf_workers: int,
    owner_cpu_budget: Optional[int],
    task_count: int,
) -> int:
    """Validate the thread budget for independent top-level nuisance folds."""

    if isinstance(tfidf_workers, bool) or not isinstance(tfidf_workers, int):
        raise TypeError("tfidf_workers must be a positive integer")
    if int(tfidf_workers) < 1:
        raise ValueError("tfidf_workers must be a positive integer")
    budget = int(tfidf_workers)
    if owner_cpu_budget is not None:
        if isinstance(owner_cpu_budget, bool) or not isinstance(owner_cpu_budget, int):
            raise TypeError("TF-IDF owner_cpu_budget must be a positive integer")
        if int(owner_cpu_budget) < 1:
            raise ValueError("TF-IDF owner_cpu_budget must be a positive integer")
        budget = int(owner_cpu_budget)
    if isinstance(task_count, bool) or int(task_count) < 1:
        raise ValueError("TF-IDF nuisance top-level fold count must be positive")
    return min(int(tfidf_workers), budget, int(task_count))


def fit_joint_cross_fitted_nuisance_stacks(
    *,
    texts: Sequence[str],
    treatment: Sequence[float],
    outcome: Sequence[float],
    outcome_binary: bool,
    strata: Sequence[int],
    views: Sequence[BoWViewConfig],
    folds: int,
    random_state: int,
    nuisance_stack_config: TfidfNuisanceStackScientificConfig,
    tfidf_workers: int = 1,
    owner_cpu_budget: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """Fit treatment/outcome stacks while sharing every label-free TF-IDF fit.

    Independent top-level folds may run on ``tfidf_workers`` threads; every
    fold is seeded from its fold coordinates, so results are identical to the
    serial loop.
    """
    if type(nuisance_stack_config) is not TfidfNuisanceStackScientificConfig:
        raise TypeError("nuisance_stack_config must be explicitly typed")
    texts = _normalize_texts(texts)
    targets = {
        "treatment": (np.asarray(treatment, dtype=float), True),
        "outcome": (np.asarray(outcome, dtype=float), bool(outcome_binary)),
    }
    if outcome_binary:
        targets.update(
            mu0=(np.asarray(outcome, float), True), mu1=(np.asarray(outcome, float), True)
        )

    def target_rows(target):
        return (
            np.asarray(treatment) == int(target[-1])
            if target in {"mu0", "mu1"}
            else np.ones(len(texts), bool)
        )

    split_labels = np.asarray(strata, dtype=int)
    views = list(views)
    n_rows = len(texts)
    fold_count = _bounded_folds(folds, split_labels, stratified=True)
    base_oof = {target: np.full((n_rows, len(views)), np.nan, dtype=float) for target in targets}
    stack_oof = {target: np.full(n_rows, np.nan, dtype=float) for target in targets}
    fold_ids = np.full(n_rows, -1, dtype=int)
    fit_positions_by_row: List[List[int]] = [[] for _ in range(n_rows)]
    grouped: Dict[Tuple[Any, ...], List[Tuple[int, BoWViewConfig]]] = {}
    for index, view in enumerate(views):
        grouped.setdefault(_vectorizer_key(view), []).append((index, view))

    tasks = [
        _JointNuisanceTopLevelFoldTask(
            fold=int(fold),
            fit_positions=np.asarray(fit_pos, dtype=int),
            heldout_positions=np.asarray(heldout_pos, dtype=int),
            texts=tuple(texts),
            treatment=np.asarray(targets["treatment"][0], dtype=float),
            outcome=np.asarray(targets["outcome"][0], dtype=float),
            outcome_binary=bool(outcome_binary),
            split_labels=split_labels,
            views=tuple(views),
            fold_count=int(fold_count),
            random_state=int(random_state),
            nuisance_stack_config=nuisance_stack_config,
        )
        for fold, (fit_pos, heldout_pos) in enumerate(
            _splitter(
                split_labels,
                fold_count,
                stratified=True,
                seed=random_state,
                shuffle=nuisance_stack_config.split_shuffle,
            ),
            start=1,
        )
    ]
    workers = _joint_nuisance_workers(
        tfidf_workers=tfidf_workers,
        owner_cpu_budget=owner_cpu_budget,
        task_count=len(tasks),
    )
    with threadpool_limits(limits=1):
        if workers == 1:
            fold_results = [_fit_joint_nuisance_top_level_fold(task) for task in tasks]
        else:
            with parallel_config(backend="threading", n_jobs=workers):
                fold_results = Parallel(batch_size=1, pre_dispatch="all")(
                    delayed(_fit_joint_nuisance_top_level_fold)(task) for task in tasks
                )
    for result in fold_results:
        fold = int(result.fold)
        fit_pos = np.asarray(result.fit_positions, dtype=int)
        heldout_pos = np.asarray(result.heldout_positions, dtype=int)
        if np.any(fold_ids[heldout_pos] != -1):
            raise RuntimeError("TF-IDF nuisance folds overlap held-out positions")
        for target in targets:
            base_oof[target][heldout_pos] = np.asarray(
                result.base_predictions[target],
                dtype=float,
            )
            stack_oof[target][heldout_pos] = np.asarray(
                result.stacked_predictions[target],
                dtype=float,
            )
        fold_ids[heldout_pos] = fold
        fit_list = fit_pos.astype(int).tolist()
        for position in heldout_pos:
            fit_positions_by_row[int(position)] = fit_list
    if np.any(fold_ids < 1):
        raise RuntimeError("TF-IDF nuisance merge omitted held-out positions")

    fitted_bases: Dict[str, List[Tuple[Optional[TfidfVectorizer], Optional[Any], float]]] = {
        target: [(None, None, float(np.mean(values))) for _ in views]
        for target, (values, _binary) in targets.items()
    }
    for group_index, group_views in enumerate(grouped.values()):
        vectorizer = _vectorizer(group_views[0][1])
        try:
            x_full = vectorizer.fit_transform(texts)
        except ValueError as exc:
            if group_views[0][1].empty_vocabulary_policy == "fail_closed":
                raise ValueError(
                    "joint nuisance full fit could not fit the configured "
                    f"vocabulary for view {group_views[0][1].name!r}"
                ) from exc
            vectorizer = None
            x_full = None
        for view_index, view in group_views:
            for target_index, (target, (values, binary)) in enumerate(targets.items()):
                mask = target_rows(target)
                model = None
                if x_full is not None and not (
                    binary and len(np.unique(values[mask].astype(int))) < 2
                ):
                    _, model = _matrix_model_prediction(
                        x_full[mask],
                        x_full[:0],
                        values[mask],
                        view,
                        binary=binary,
                        seed=random_state
                        + 50_000
                        + 100 * group_index
                        + 10 * view_index
                        + target_index,
                    )
                fitted_bases[target][view_index] = (
                    vectorizer,
                    model,
                    float(np.mean(values[mask])),
                )

    output: Dict[str, Dict[str, Any]] = {}
    for target_index, (target, (values, binary)) in enumerate(targets.items()):
        if not np.isfinite(base_oof[target]).all() or not np.isfinite(stack_oof[target]).all():
            raise RuntimeError(f"Cross-fitted {target} nuisance predictions are incomplete")
        mask = target_rows(target)
        complete_stack, complete_constant = _fit_stack(
            base_oof[target][mask],
            values[mask],
            binary=binary,
            seed=random_state + 60_000 + target_index,
            config=nuisance_stack_config,
        )
        fitted = CrossFittedStack(
            views=views,
            binary=binary,
            base_models=fitted_bases[target],
            stack_model=complete_stack,
            stack_constant=complete_constant,
            config_hash=stable_hash(
                {
                    "views": [asdict(view) for view in views],
                    "folds": fold_count,
                    "target": target,
                    "joint_label_free_vectorization": True,
                    "random_state": random_state,
                    "nuisance_stack_scientific": asdict(
                        nuisance_stack_config
                    ),
                }
            ),
        )
        output[target] = {
            "base_oof": base_oof[target],
            "stacked_oof": stack_oof[target],
            "fold_ids": fold_ids.copy(),
            "fit_positions_by_row": [list(values) for values in fit_positions_by_row],
            "fitted": fitted,
            "metrics": nuisance_metrics(
                values[mask],
                stack_oof[target][mask],
                binary=binary,
                calibration_config=nuisance_stack_config.calibration,
            ),
            "view_metrics": {
                view.name: nuisance_metrics(
                    values[mask],
                    base_oof[target][mask, index],
                    binary=binary,
                    calibration_config=nuisance_stack_config.calibration,
                )
                for index, view in enumerate(views)
            },
        }
    return output


def calibration_intercept_slope(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    config: NuisanceCalibrationScientificConfig,
) -> Tuple[Any, Any]:
    labels = np.asarray(labels, dtype=int)
    epsilon = float(config.probability_clip_epsilon)
    probabilities = np.clip(
        np.asarray(probabilities, dtype=float),
        epsilon,
        1.0 - epsilon,
    )
    if len(np.unique(labels)) < 2:
        return None, None
    logits = np.log(probabilities / (1.0 - probabilities)).reshape(-1, 1)
    model = _configured_logistic_regression(
        c=config.logistic_c,
        max_iter=config.logistic_max_iter,
        config=config.logistic,
        seed=0,
    )
    try:
        model.fit(logits, labels)
    except ValueError:
        return None, None
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    config: NuisanceCalibrationScientificConfig,
) -> float:
    labels = np.asarray(labels, dtype=float)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    n_bins = int(config.ece_bins)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    bins = np.minimum(np.digitize(probabilities, edges[1:-1]), n_bins - 1)
    error = 0.0
    for index in range(n_bins):
        mask = bins == index
        if np.any(mask):
            error += float(np.mean(mask)) * abs(
                float(np.mean(labels[mask])) - float(np.mean(probabilities[mask]))
            )
    return float(error)


def nuisance_metrics(
    values: np.ndarray,
    predictions: np.ndarray,
    *,
    binary: bool,
    calibration_config: NuisanceCalibrationScientificConfig,
) -> Dict[str, Any]:
    values = np.asarray(values, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    if not binary:
        return {
            "rmse": float(np.sqrt(mean_squared_error(values, predictions))),
            "mae": float(mean_absolute_error(values, predictions)),
        }
    epsilon = float(calibration_config.probability_clip_epsilon)
    probabilities = np.clip(predictions, epsilon, 1.0 - epsilon)
    try:
        auroc = float(roc_auc_score(values.astype(int), probabilities))
    except ValueError:
        auroc = None
    intercept, slope = calibration_intercept_slope(
        values,
        probabilities,
        config=calibration_config,
    )
    return {
        "auroc": auroc,
        "brier": float(brier_score_loss(values.astype(int), probabilities)),
        "log_loss": float(log_loss(values.astype(int), probabilities, labels=[0, 1])),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "ece": expected_calibration_error(
            values,
            probabilities,
            config=calibration_config,
        ),
    }


def cohort_contrast_scores(
    matrix: Any,
    feature_names: Sequence[str],
    treatment: Sequence[float],
    outcome: Sequence[float],
    propensity_prediction: Sequence[float],
    outcome_prediction: Sequence[float],
    *,
    dina_nuisance=None,
) -> pd.DataFrame:
    """Calculate signed orthogonal cohort moments and robust standard errors."""
    x = sparse.csr_matrix(matrix, dtype=float)
    names = np.asarray(feature_names, dtype=object)
    t = np.asarray(treatment, dtype=float)
    y = np.asarray(outcome, dtype=float)
    e = np.asarray(propensity_prediction, dtype=float)
    m = np.asarray(outcome_prediction, dtype=float)
    if x.shape != (len(t), len(names)):
        raise ValueError("matrix shape must match rows and feature_names")
    u = t - e
    v = y - m
    denominator = float(np.dot(u, u))
    constant_effect = 0.0 if denominator <= 0.0 else float(np.dot(u, v) / denominator)
    contribution = u * (v - constant_effect * u)
    centers = np.zeros(x.shape[1])
    if dina_nuisance is not None:
        from ..models import dina

        contribution, weights, constant_effect = dina.score(
            y, t, dina_nuisance["a"], dina_nuisance["nu"]
        )
        centers = np.asarray(x.T @ weights).ravel() / max(weights.sum(), 1e-15)
    n_rows = max(1, len(t))
    moments = (np.asarray(x.T @ contribution).ravel() - centers * contribution.sum()) / n_rows
    squared_sum = np.asarray(x.power(2).T @ np.square(contribution)).ravel()
    squared_sum += (
        -2 * centers * np.asarray(x.T @ np.square(contribution)).ravel()
        + centers**2 * np.square(contribution).sum()
    )
    variances = np.maximum(0.0, squared_sum / n_rows - np.square(moments))
    robust_se = np.sqrt(variances / n_rows)
    scores = np.divide(
        moments,
        robust_se,
        out=np.zeros_like(moments),
        where=robust_se > 0.0,
    )
    presence = x.copy()
    presence.data = np.ones_like(presence.data)
    support_control = np.asarray(presence[t < 0.5].sum(axis=0)).ravel().astype(int)
    support_treated = np.asarray(presence[t >= 0.5].sum(axis=0)).ravel().astype(int)
    return pd.DataFrame(
        {
            "feature": names.astype(str),
            "moment": moments,
            "robust_se": robust_se,
            "signed_score": scores,
            "unsigned_score": np.abs(scores),
            "support_control": support_control,
            "support_treated": support_treated,
            "constant_residual_effect": constant_effect,
            "effect_objective": (
                "bernoulli_dina_score" if dina_nuisance is not None else "orthogonal_r_score"
            ),
        }
    )


def unsigned_linear_screen(
    matrix: Any,
    feature_names: Sequence[str],
    values: Sequence[float],
    *,
    binary: bool,
    view: BoWViewConfig,
    random_state: int,
) -> pd.DataFrame:
    """Return unsigned selection importance while preserving coefficient sign."""
    if type(view) is not BoWViewConfig or view.bow_model != "linear":
        raise ValueError("linear screen requires an explicit configured linear view")
    x = sparse.csr_matrix(matrix)
    values = np.asarray(values, dtype=float)
    if binary:
        if len(np.unique(values.astype(int))) < 2:
            coefficients = np.zeros(x.shape[1], dtype=float)
        else:
            model = _classifier(view, random_state).fit(
                x,
                values.astype(int),
            )
            coefficients = np.asarray(model.coef_).reshape(-1)
    else:
        model = _regressor(view, random_state).fit(x, values)
        coefficients = np.asarray(model.coef_).reshape(-1)
    frame = pd.DataFrame(
        {
            "feature": np.asarray(feature_names, dtype=str),
            "signed_score": coefficients,
            "unsigned_score": np.abs(coefficients),
        }
    )
    return frame.sort_values(
        ["unsigned_score", "feature"], ascending=[False, True], ignore_index=True
    )


def _subsample_indices(strata: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    selected: List[int] = []
    for value in np.unique(strata):
        positions = np.where(strata == value)[0]
        count = max(1, min(len(positions), int(np.ceil(len(positions) * fraction))))
        selected.extend(rng.choice(positions, size=count, replace=False).astype(int).tolist())
    return np.asarray(sorted(selected), dtype=int)


def add_linear_stability(
    screen: pd.DataFrame,
    matrix: Any,
    values: np.ndarray,
    *,
    binary: bool,
    strata: np.ndarray,
    config: TfidfTopicDiscoveryConfig,
    view: BoWViewConfig,
    random_state: int,
) -> pd.DataFrame:
    result = screen.copy()
    n_features = len(result)
    feature_to_index = {name: index for index, name in enumerate(result["feature"])}
    selection = np.zeros(n_features, dtype=float)
    rank_score = np.zeros(n_features, dtype=float)
    signs = np.zeros(n_features, dtype=float)
    repeats = int(config.stability_repeats)
    rng = np.random.default_rng(random_state)
    top_count = max(1, int(np.ceil(config.top_fraction * n_features)))
    for repeat in range(repeats):
        positions = _subsample_indices(strata, config.stability_fraction, rng)
        sample = unsigned_linear_screen(
            sparse.csr_matrix(matrix)[positions],
            result["feature"].tolist(),
            values[positions],
            binary=binary,
            view=view,
            random_state=random_state + repeat + 1,
        )
        for rank, row in sample.iterrows():
            index = feature_to_index[str(row["feature"])]
            if rank < top_count:
                selection[index] += 1.0
            rank_score[index] += 1.0 - rank / max(1, n_features - 1)
            signs[index] += np.sign(float(row["signed_score"]))
    divisor = max(1, repeats)
    result["selection_stability"] = selection / divisor
    result["rank_stability"] = rank_score / divisor
    result["sign_stability"] = np.abs(signs) / divisor
    screening = config.screening_scientific
    result["combined_importance"] = result["unsigned_score"] * (
        float(screening.linear_base_weight)
        + float(screening.linear_selection_stability_weight)
        * result["selection_stability"]
        + float(screening.linear_rank_stability_weight)
        * result["rank_stability"]
    )
    return result.sort_values(
        ["combined_importance", "unsigned_score", "feature"],
        ascending=[False, False, True],
        ignore_index=True,
    )


def add_effect_stability(
    scores: pd.DataFrame,
    matrix: Any,
    treatment: np.ndarray,
    outcome: np.ndarray,
    stacked_e: np.ndarray,
    stacked_m: np.ndarray,
    nuisance_sources: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    strata: np.ndarray,
    config: TfidfTopicDiscoveryConfig,
    random_state: int,
    dina_nuisance=None,
) -> pd.DataFrame:
    result = scores.copy()
    x = sparse.csr_matrix(matrix)
    names = result["feature"].tolist()
    reference_sign = np.sign(result["signed_score"].to_numpy(dtype=float))
    source_agreements: List[np.ndarray] = []
    for source_values in nuisance_sources:
        source_e, source_m = source_values[:2]
        dn = None
        if dina_nuisance is not None:
            from ..models import dina

            dn = dina.nuisances(
                source_e,
                *(
                    source_values[2:]
                    if len(source_values) == 4
                    else (dina_nuisance["mu0"], dina_nuisance["mu1"])
                ),
            )
        source = cohort_contrast_scores(
            x, names, treatment, outcome, source_e, source_m, dina_nuisance=dn
        )["signed_score"].to_numpy(dtype=float)
        source_agreements.append((np.sign(source) == reference_sign).astype(float))
    result["nuisance_source_agreement"] = (
        np.mean(np.vstack(source_agreements), axis=0)
        if source_agreements
        else np.ones(len(result), dtype=float)
    )

    repeats = int(config.stability_repeats)
    selection = np.zeros(len(result), dtype=float)
    signs = np.zeros(len(result), dtype=float)
    rng = np.random.default_rng(random_state)
    eligible_support = (
        result["support_control"].to_numpy() >= config.minimum_arm_document_support
    ) & (result["support_treated"].to_numpy() >= config.minimum_arm_document_support)
    eligible_count = max(1, int(np.sum(eligible_support)))
    top_count = max(1, int(np.ceil(config.top_fraction * eligible_count)))
    for _ in range(repeats):
        positions = _subsample_indices(strata, config.stability_fraction, rng)
        sample = cohort_contrast_scores(
            x[positions],
            names,
            treatment[positions],
            outcome[positions],
            stacked_e[positions],
            stacked_m[positions],
            dina_nuisance=(
                None
                if dina_nuisance is None
                else {k: v[positions] for k, v in dina_nuisance.items()}
            ),
        )
        magnitude = sample["unsigned_score"].to_numpy(dtype=float)
        valid = np.where(eligible_support)[0]
        chosen = valid[np.argsort(-magnitude[valid], kind="stable")[:top_count]]
        selection[chosen] += 1.0
        signs += (np.sign(sample["signed_score"].to_numpy(dtype=float)) == reference_sign).astype(
            float
        )
    divisor = max(1, repeats)
    result["subsample_selection_stability"] = selection / divisor
    result["subsample_sign_agreement"] = signs / divisor

    contribution = (treatment - stacked_e) * (
        (outcome - stacked_m)
        - float(result["constant_residual_effect"].iloc[0]) * (treatment - stacked_e)
    )
    if dina_nuisance is not None:
        from ..models import dina

        contribution, _, _ = dina.score(
            outcome,
            treatment,
            dina_nuisance["a"],
            dina_nuisance["nu"],
            delta=float(result["constant_residual_effect"].iloc[0]),
        )
    tail_agreement = np.zeros(len(result), dtype=float)
    csc = x.tocsc()
    for index in range(x.shape[1]):
        values = np.asarray(csc[:, index].toarray()).ravel()
        present = values > 0.0
        if np.any(present) and np.any(~present):
            tail_sign = np.sign(np.mean(contribution[present]) - np.mean(contribution[~present]))
            tail_agreement[index] = float(tail_sign == reference_sign[index])
    result["tail_contrast_sign_agreement"] = tail_agreement
    result["eligible"] = (
        eligible_support
        & (result["nuisance_source_agreement"] >= config.minimum_nuisance_source_agreement)
        & (result["subsample_selection_stability"] >= config.minimum_subsample_selection_fraction)
        & (result["tail_contrast_sign_agreement"] >= config.minimum_tail_sign_agreement)
    )
    screening = config.screening_scientific
    result["combined_importance"] = result["unsigned_score"] * (
        float(screening.effect_base_weight)
        + float(screening.effect_nuisance_agreement_weight)
        * result["nuisance_source_agreement"]
        + float(screening.effect_selection_stability_weight)
        * result["subsample_selection_stability"]
        + float(screening.effect_sign_stability_weight)
        * result["subsample_sign_agreement"]
    )
    return result.sort_values(
        ["eligible", "combined_importance", "unsigned_score", "feature"],
        ascending=[False, False, False, True],
        ignore_index=True,
    )


def align_topic_components(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Return candidate indices aligned to reference topics by cosine/Hungarian match."""
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError("reference and candidate topic matrices must have equal 2D shapes")
    reference_norm = np.linalg.norm(reference, axis=1, keepdims=True)
    candidate_norm = np.linalg.norm(candidate, axis=1, keepdims=True)
    reference_unit = reference / np.where(reference_norm > 0.0, reference_norm, 1.0)
    candidate_unit = candidate / np.where(candidate_norm > 0.0, candidate_norm, 1.0)
    reference_rows, candidate_rows = linear_sum_assignment(-(reference_unit @ candidate_unit.T))
    permutation = np.empty(reference.shape[0], dtype=int)
    permutation[reference_rows] = candidate_rows
    return permutation


@dataclass
class ConsensusNMFTopicBank:
    bank_name: str
    feature_names: List[str]
    selected_indices: np.ndarray
    feature_weights: np.ndarray
    alignments: List[np.ndarray]
    consensus_loadings: np.ndarray
    topic_terms: List[List[Dict[str, Any]]]
    requested_components: int
    actual_components: int
    terms_per_topic: int
    seeds: List[int]
    reduction_reason: Optional[str] = None

    @classmethod
    def fit(
        cls,
        *,
        bank_name: str,
        matrix: Any,
        feature_names: Sequence[str],
        scores: pd.DataFrame,
        config: TfidfTopicDiscoveryConfig,
    ) -> Optional["ConsensusNMFTopicBank"]:
        x = sparse.csr_matrix(matrix, dtype=np.float64)
        names = np.asarray(feature_names, dtype=str)
        score_by_name = scores.set_index("feature")
        eligible = scores
        if "eligible" in eligible.columns:
            eligible = eligible[eligible["eligible"].astype(bool)]
        if eligible.empty or x.shape[1] == 0 or x.shape[0] < 2:
            return None
        target_count = max(
            min(config.topic_count, len(eligible)),
            min(config.terms_per_topic, len(eligible)),
        )
        select_count = min(
            len(eligible),
            max(target_count, int(np.ceil(config.top_fraction * len(eligible)))),
        )
        selected_names = eligible.head(select_count)["feature"].astype(str).tolist()
        index_by_name = {name: index for index, name in enumerate(names)}
        selected_indices = np.asarray(
            [index_by_name[name] for name in selected_names if name in index_by_name], dtype=int
        )
        if len(selected_indices) == 0:
            return None
        importance = np.asarray(
            [
                float(score_by_name.loc[names[index], "combined_importance"])
                for index in selected_indices
            ],
            dtype=float,
        )
        positive = importance[np.isfinite(importance) & (importance > 0.0)]
        median = float(np.median(positive)) if len(positive) else 1.0
        weights = np.clip(
            np.sqrt(
                np.maximum(importance, 0.0)
                / max(
                    median,
                    float(config.nmf_scientific.importance_scale_epsilon),
                )
            ),
            config.importance_weight_min,
            config.importance_weight_max,
        )
        weighted = x[:, selected_indices].multiply(weights).tocsr()
        actual = min(config.topic_count, x.shape[0] - 1, len(selected_indices))
        if actual < 1:
            return None
        reason = None
        if actual < config.topic_count:
            reason = (
                f"requested={config.topic_count}; rows={x.shape[0]}; "
                f"selected_terms={len(selected_indices)}"
            )
        normalized_h: List[np.ndarray] = []
        for seed in config.topic_seeds:
            model = NMF(
                n_components=actual,
                init="nndsvdar",
                solver="cd",
                beta_loss="frobenius",
                max_iter=config.nmf_max_iter,
                tol=config.nmf_tol,
                random_state=int(seed),
                alpha_W=float(config.nmf_scientific.alpha_w),
                alpha_H=config.nmf_scientific.alpha_h,
                l1_ratio=float(config.nmf_scientific.l1_ratio),
                verbose=0,
                shuffle=bool(config.nmf_scientific.shuffle),
            )
            fit_transform_nmf_with_c_order_initial_w(model, weighted)
            h = np.asarray(model.components_, dtype=float)
            component_norm = np.linalg.norm(h, axis=1)
            component_norm = np.where(component_norm > 0.0, component_norm, 1.0)
            normalized_h.append(h / component_norm[:, None])
        reference = normalized_h[0]
        alignments: List[np.ndarray] = [np.arange(actual, dtype=int)]
        aligned_h = [reference]
        for index in range(1, len(normalized_h)):
            permutation = align_topic_components(reference, normalized_h[index])
            alignments.append(permutation)
            aligned_h.append(normalized_h[index][permutation])
        consensus_h = np.mean(np.stack(aligned_h), axis=0)

        selected_name_array = names[selected_indices]
        all_ranked_names = scores["feature"].astype(str).tolist()
        terms: List[List[Dict[str, Any]]] = []
        for topic_index in range(actual):
            order = np.argsort(-consensus_h[topic_index], kind="stable")
            topic_rows: List[Dict[str, Any]] = []
            used = set()
            for selected_position in order:
                name = str(selected_name_array[selected_position])
                used.add(name)
                topic_rows.append(
                    {
                        "term": name,
                        "loading": float(consensus_h[topic_index, selected_position]),
                        "screen_rank": int(score_by_name.index.get_loc(name)) + 1,
                        "signed_score": float(score_by_name.loc[name, "signed_score"]),
                    }
                )
                if len(topic_rows) == config.terms_per_topic:
                    break
            # A selected set smaller than the configured evidence capacity can
            # still yield a complete traceable prompt by appending the
            # highest-ranked zero-loading vocabulary.
            for name in all_ranked_names:
                if len(topic_rows) == config.terms_per_topic:
                    break
                if name in used:
                    continue
                used.add(name)
                topic_rows.append(
                    {
                        "term": name,
                        "loading": 0.0,
                        "screen_rank": int(score_by_name.index.get_loc(name)) + 1,
                        "signed_score": float(score_by_name.loc[name, "signed_score"]),
                    }
                )
            if len(topic_rows) != int(config.terms_per_topic):
                raise ValueError(
                    f"{bank_name} topic {topic_index + 1} could account for only "
                    f"{len(topic_rows)} of configured terms_per_topic="
                    f"{int(config.terms_per_topic)}; refusing a partial topic "
                    "evidence record"
                )
            terms.append(topic_rows)
        bank = cls(
            bank_name=bank_name,
            feature_names=names.tolist(),
            selected_indices=selected_indices,
            feature_weights=weights,
            alignments=alignments,
            consensus_loadings=consensus_h,
            topic_terms=terms,
            requested_components=int(config.topic_count),
            actual_components=int(actual),
            terms_per_topic=int(config.terms_per_topic),
            seeds=list(config.topic_seeds),
            reduction_reason=reason,
        )
        return bank

    def metadata(self) -> Dict[str, Any]:
        if (
            int(self.terms_per_topic) < 1
            or any(
                len(topic_terms) != int(self.terms_per_topic)
                for topic_terms in self.topic_terms
            )
        ):
            raise RuntimeError(
                "fitted topic bank does not contain the configured complete "
                "term evidence"
            )
        return {
            "bank": self.bank_name,
            "requested_topic_count": self.requested_components,
            "actual_topic_count": self.actual_components,
            "terms_per_topic": int(self.terms_per_topic),
            "component_reduction_reason": self.reduction_reason,
            "seeds": self.seeds,
            "selected_term_count": int(len(self.selected_indices)),
            "selected_terms": [self.feature_names[index] for index in self.selected_indices],
            "feature_weights": self.feature_weights.astype(float).tolist(),
            "alignments": [alignment.astype(int).tolist() for alignment in self.alignments],
            "topics": [
                {
                    "topic_id": f"{self.bank_name}_topic_{index + 1:03d}",
                    "bank": self.bank_name,
                    "terms_per_topic": int(self.terms_per_topic),
                    "terms": topic_terms,
                }
                for index, topic_terms in enumerate(self.topic_terms)
            ],
        }


def _common_vectorizer(config: TfidfTopicDiscoveryConfig) -> TfidfVectorizer:
    return _make_bow_vectorizer(asdict(config.vectorizer_scientific))


def _strata(treatment: np.ndarray, outcome: np.ndarray, *, outcome_binary: bool) -> np.ndarray:
    if outcome_binary:
        outcome_group = outcome.astype(int)
    else:
        try:
            outcome_group = pd.qcut(
                outcome, q=min(4, len(np.unique(outcome))), labels=False, duplicates="drop"
            ).to_numpy()
        except AttributeError:
            outcome_group = np.asarray(
                pd.qcut(outcome, q=min(4, len(np.unique(outcome))), labels=False, duplicates="drop")
            )
        outcome_group = np.nan_to_num(outcome_group, nan=0).astype(int)
    return treatment.astype(int) * (int(np.max(outcome_group)) + 1) + outcome_group


def compact_orphan_ngram_branch(orphan: Mapping[str, Any]) -> Dict[str, Any]:
    """Build the small handoff view of the orphan n-gram cluster selection."""
    return {
        key: orphan.get(key)
        for key in (
            "status",
            "candidate_definition",
            "topic_term_exclusion_is_fit_side",
            "cluster_construction_uses_heldout_rows_or_labels",
            "candidate_count_before_topic_exclusion",
            "represented_topic_term_exclusion_count",
            "candidate_count_before_nested_deduplication",
            "deduplicated_alias_count",
            "representative_count",
            "cluster_count",
            "selected_cluster_ids",
            "selected_clusters",
            "selection_count",
            "selection_rule",
            "minimum_selected_clusters",
            "maximum_selected_clusters",
        )
    }


def fit_tfidf_topic_context(
    *,
    fit_df: pd.DataFrame,
    heldout_df: pd.DataFrame,
    text_column: str,
    treatment_column: str,
    outcome_column: str,
    outcome_type: str,
    views: Sequence[BoWViewConfig],
    nuisance_folds: int,
    config: TfidfTopicDiscoveryConfig,
    artifact_dir: Path,
    scope_id: str,
    score_orphan_ngrams: bool = False,
    timer: Any = None,
) -> Tuple[Dict[str, Any], Dict[str, CrossFittedStack]]:
    """Fit one exact context and predict its externally held-out row set.

    Returns the context metadata and the fitted treatment/outcome nuisance
    stacks.  Held-out labels are read only when ``score_orphan_ngrams`` is
    true, in which case ``heldout_df`` must be a nested calibration partition
    of the registered fit rows rather than a registered held-out set.
    ``timer`` (reporting only) receives one lap per section.
    """
    from .stage1_architectures import TFIDF_ORPHAN_NGRAMS, TFIDF_TOPICS
    from .stage1_timing import SHARED_ATTRIBUTION, TFIDF_ARCHITECTURES, NullTimer

    timer = timer or NullTimer()
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    fit_texts = _normalize_texts(fit_df[text_column].fillna(""))
    heldout_texts = _normalize_texts(heldout_df[text_column].fillna(""))
    treatment = fit_df[treatment_column].to_numpy(dtype=float)
    outcome = fit_df[outcome_column].to_numpy(dtype=float)
    outcome_binary = str(outcome_type).lower() != "continuous"
    strata = _strata(treatment, outcome, outcome_binary=outcome_binary)
    joint_nuisance = fit_joint_cross_fitted_nuisance_stacks(
        texts=fit_texts,
        treatment=treatment,
        outcome=outcome,
        outcome_binary=outcome_binary,
        strata=strata,
        views=views,
        folds=nuisance_folds,
        random_state=config.random_state + 101,
        nuisance_stack_config=config.nuisance_stack_scientific,
    )
    treatment_result = joint_nuisance["treatment"]
    outcome_result = joint_nuisance["outcome"]
    external_e, external_e_views = treatment_result["fitted"].predict(heldout_texts)
    external_m, external_m_views = outcome_result["fitted"].predict(heldout_texts)
    dn = dv = None
    if outcome_binary:
        from ..models import dina

        dn = dina.nuisances(
            treatment_result["stacked_oof"],
            joint_nuisance["mu0"]["stacked_oof"],
            joint_nuisance["mu1"]["stacked_oof"],
        )
        dv = dina.nuisances(
            external_e,
            joint_nuisance["mu0"]["fitted"].predict(heldout_texts)[0],
            joint_nuisance["mu1"]["fitted"].predict(heldout_texts)[0],
        )
    timer.lap(
        "tfidf_nuisance_stacks",
        attribution=SHARED_ATTRIBUTION,
        consumers=TFIDF_ARCHITECTURES,
    )

    vectorizer = _common_vectorizer(config)
    common_fit = vectorizer.fit_transform(fit_texts)
    feature_names = vectorizer.get_feature_names_out()

    screening_view_name = config.screening_scientific.model_source_view_name
    matching_views = [
        view for view in views if view.name == screening_view_name
    ]
    if len(matching_views) != 1:
        raise ValueError(
            "TF-IDF screening requires exactly one explicitly configured "
            f"BoW view named {screening_view_name!r}"
        )
    linear_view = matching_views[0]
    if (
        linear_view.bow_model != "linear"
        or linear_view.ngram_range_min != 1
        or linear_view.ngram_range_max != 3
    ):
        raise ValueError(
            "TF-IDF screening source must be a configured linear 1-3-gram view"
        )
    treatment_scores = add_linear_stability(
        unsigned_linear_screen(
            common_fit,
            feature_names,
            treatment,
            binary=True,
            view=linear_view,
            random_state=config.random_state + 301,
        ),
        common_fit,
        treatment,
        binary=True,
        strata=strata,
        config=config,
        view=linear_view,
        random_state=config.random_state + 311,
    )
    treatment_scores["eligible"] = True
    outcome_scores = add_linear_stability(
        unsigned_linear_screen(
            common_fit,
            feature_names,
            outcome,
            binary=outcome_binary,
            view=linear_view,
            random_state=config.random_state + 401,
        ),
        common_fit,
        outcome,
        binary=outcome_binary,
        strata=strata,
        config=config,
        view=linear_view,
        random_state=config.random_state + 411,
    )
    outcome_scores["eligible"] = True

    nuisance_sources = [
        (treatment_result["base_oof"][:, index], outcome_result["base_oof"][:, index])
        + (
            (
                joint_nuisance["mu0"]["base_oof"][:, index],
                joint_nuisance["mu1"]["base_oof"][:, index],
            )
            if outcome_binary
            else ()
        )
        for index in range(
            min(treatment_result["base_oof"].shape[1], outcome_result["base_oof"].shape[1])
        )
    ]
    effect_scores = add_effect_stability(
        cohort_contrast_scores(
            common_fit,
            feature_names,
            treatment,
            outcome,
            treatment_result["stacked_oof"],
            outcome_result["stacked_oof"],
            dina_nuisance=dn,
        ),
        common_fit,
        treatment,
        outcome,
        treatment_result["stacked_oof"],
        outcome_result["stacked_oof"],
        nuisance_sources,
        strata=strata,
        config=config,
        random_state=config.random_state + 501,
        dina_nuisance=dn,
    )

    score_frames = {
        "treatment": treatment_scores,
        "outcome": outcome_scores,
        "effect": effect_scores,
    }
    bank_metadata: Dict[str, Dict[str, Any]] = {}
    score_paths: Dict[str, str] = {}
    for bank_name, score_frame in score_frames.items():
        eligible_count = int(
            score_frame["eligible"].astype(bool).sum()
            if "eligible" in score_frame
            else len(score_frame)
        )
        score_path = artifact_dir / f"{bank_name}_ngram_scores.parquet"
        score_frame.to_parquet(score_path, index=False)
        score_paths[bank_name] = str(score_path)
        bank = ConsensusNMFTopicBank.fit(
            bank_name=bank_name,
            matrix=common_fit,
            feature_names=feature_names,
            scores=score_frame,
            config=config,
        )
        if bank is None:
            bank_metadata[bank_name] = {
                "bank": bank_name,
                "requested_topic_count": config.topic_count,
                "actual_topic_count": 0,
                "terms_per_topic": int(config.terms_per_topic),
                "weak_or_unstable_raw_evidence": bank_name == "effect",
                "eligible_candidate_count": eligible_count,
                "selected_candidate_count": 0,
                "discarded_candidate_count": eligible_count,
                "selection_rule": (
                    "eligible_combined_importance_then_configured_top_fraction"
                ),
                "topics": [],
            }
            continue
        bank_metadata[bank_name] = bank.metadata()
        bank_metadata[bank_name]["weak_or_unstable_raw_evidence"] = False
        selected_count = int(len(bank.selected_indices))
        bank_metadata[bank_name].update(
            {
                "eligible_candidate_count": eligible_count,
                "selected_candidate_count": selected_count,
                "discarded_candidate_count": max(
                    0,
                    eligible_count - selected_count,
                ),
                "selection_rule": (
                    "eligible_combined_importance_then_configured_top_fraction"
                ),
            }
        )

    timer.lap("tfidf_topic_screens_and_banks", attribution=TFIDF_TOPICS)
    orphan_branch: Optional[Dict[str, Any]] = None
    orphan_scores_path: Optional[Path] = None
    if bool(score_orphan_ngrams) and bool(config.orphan_ngram_enabled):
        # The held-out rows here are a nested calibration partition of the
        # registered fit rows; registered held-out labels are never read.
        orphan_branch = score_effect_orphan_ngram_clusters(
            dina_nuisance=(dn, dv) if dn is not None else None,
            fit_matrix=sparse.csr_matrix(common_fit),
            heldout_matrix=sparse.csr_matrix(vectorizer.transform(heldout_texts)),
            feature_names=feature_names,
            effect_scores=score_frames["effect"],
            effect_topics=list((bank_metadata.get("effect") or {}).get("topics") or []),
            fit_treatment=treatment,
            fit_outcome=outcome,
            heldout_treatment=heldout_df[treatment_column].to_numpy(dtype=float),
            heldout_outcome=heldout_df[outcome_column].to_numpy(dtype=float),
            fit_propensity=np.asarray(treatment_result["stacked_oof"], dtype=float),
            fit_outcome_prediction=np.asarray(outcome_result["stacked_oof"], dtype=float),
            heldout_propensity=np.asarray(external_e, dtype=float),
            heldout_outcome_prediction=np.asarray(external_m, dtype=float),
            config=config,
        )
        orphan_scores_path = artifact_dir / "orphan_ngram_scores.json"
        orphan_scores_path.write_text(
            json.dumps(
                {"schema_version": ORPHAN_NGRAM_SCORE_SCHEMA_VERSION, **orphan_branch},
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        timer.lap("tfidf_orphan_ngram_clusters", attribution=TFIDF_ORPHAN_NGRAMS)

    fit_row_ids = fit_df["_oci_row_id"].astype(int).tolist()
    heldout_row_ids = heldout_df["_oci_row_id"].astype(int).tolist()
    oof_rows: List[Dict[str, Any]] = []
    for position, row_id in enumerate(fit_row_ids):
        row = {
            "_oci_row_id": int(row_id),
            "prediction_scope": "fit_oof",
            "nuisance_fold": int(treatment_result["fold_ids"][position]),
            "treatment_stacked": float(treatment_result["stacked_oof"][position]),
            "outcome_stacked": float(outcome_result["stacked_oof"][position]),
            "fit_row_ids": [
                int(fit_row_ids[index])
                for index in treatment_result["fit_positions_by_row"][position]
            ],
        }
        for index, view in enumerate(views):
            row[f"treatment_view__{view.name}"] = float(
                treatment_result["base_oof"][position, index]
            )
            row[f"outcome_view__{view.name}"] = float(outcome_result["base_oof"][position, index])
        if dn is not None:
            row.update({"dina_" + k: float(v[position]) for k, v in dn.items()})
        oof_rows.append(row)
    external_rows: List[Dict[str, Any]] = []
    for position, row_id in enumerate(heldout_row_ids):
        row = {
            "_oci_row_id": int(row_id),
            "prediction_scope": "external_heldout",
            "treatment_stacked": float(external_e[position]),
            "outcome_stacked": float(external_m[position]),
            "fit_row_ids": fit_row_ids,
        }
        for view in views:
            row[f"treatment_view__{view.name}"] = float(external_e_views[view.name][position])
            row[f"outcome_view__{view.name}"] = float(external_m_views[view.name][position])
        if dv is not None:
            row.update({"dina_" + k: float(v[position]) for k, v in dv.items()})
        external_rows.append(row)
    nuisance_path = artifact_dir / "nuisance_predictions.parquet"
    pd.DataFrame([*oof_rows, *external_rows]).to_parquet(nuisance_path, index=False)

    metadata = {
        "schema_version": DISCOVERY_SCHEMA_VERSION,
        "scope_id": scope_id,
        "fit_row_fingerprint": row_set_fingerprint(fit_row_ids),
        "heldout_row_fingerprint": row_set_fingerprint(heldout_row_ids),
        "fit_row_ids": fit_row_ids,
        "heldout_row_ids": heldout_row_ids,
        "config_hash": stable_hash(asdict(config)),
        "common_vocabulary_size": int(len(feature_names)),
        "common_vocabulary": feature_names.astype(str).tolist(),
        "common_vocabulary_selection": {
            "selected_term_count": int(len(feature_names)),
            "configured_max_features": config.vectorizer_scientific.max_features,
            "selection_rule": "sklearn_term_frequency_rank_v1",
            "complete_input_text_consumed": True,
        },
        "nuisance_stack_scientific": asdict(
            config.nuisance_stack_scientific
        ),
        "screening_scientific": asdict(config.screening_scientific),
        "nmf_scientific": asdict(config.nmf_scientific),
        "nuisance": {
            "treatment": {
                "stacked_metrics": treatment_result["metrics"],
                "view_metrics": treatment_result["view_metrics"],
                "stack_config_hash": treatment_result["fitted"].config_hash,
            },
            "outcome": {
                "stacked_metrics": outcome_result["metrics"],
                "view_metrics": outcome_result["view_metrics"],
                "stack_config_hash": outcome_result["fitted"].config_hash,
            },
        },
        "topic_banks": bank_metadata,
        "artifacts": {
            "ngram_scores": score_paths,
            "nuisance_predictions": str(nuisance_path),
            "orphan_ngram_scores": (
                None if orphan_scores_path is None else str(orphan_scores_path)
            ),
        },
    }
    if orphan_branch is not None:
        metadata["effect_orphan_ngram_branch"] = compact_orphan_ngram_branch(orphan_branch)
    return metadata, {
        "treatment": treatment_result["fitted"],
        "outcome": outcome_result["fitted"],
    }
