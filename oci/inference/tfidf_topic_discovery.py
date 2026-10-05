"""Honest TF-IDF nuisance modeling and three-bank topic discovery.

This module is intentionally independent of the legacy HTR, embedding, uplift,
R-learner, and raw-text causal-forest implementations.  A fitted context owns
all vectorizers, nuisance stacks, topic models, and vocabulary state needed to
transform an external held-out row set without refitting.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import stat
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
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
    TOPIC_SCORE_TEST_SCHEMA_VERSION,
    score_topic_banks,
)
from .tfidf_safe_artifacts import (
    safe_artifact_content_sha256,
    write_fitted_topic_context,
    write_named_array_bank,
)

logger = logging.getLogger(__name__)

HANDOFF_SCHEMA_VERSION = "multi_model_forest_handoff_v2"
DISCOVERY_SCHEMA_VERSION = "tfidf_topic_discovery_v3_safe_arrays_dina_v1"
TFIDF_CONTEXT_ARTIFACT_INVENTORY_SCHEMA = "tfidf_context_artifact_inventory_v1"
TFIDF_NUISANCE_EXECUTION_ATTESTATION_SCHEMA = (
    "tfidf_joint_nuisance_fold_execution_attestation_v1"
)


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def row_set_fingerprint(row_ids: Sequence[Any]) -> str:
    """Fingerprint a row set without making its input order meaningful."""
    values = sorted(str(value) for value in row_ids)
    return stable_hash(values)


def _registered_regular_file(path: Path | str, *, logical_name: str) -> Dict[str, Any]:
    requested = Path(os.path.abspath(Path(path)))
    try:
        resolved = requested.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"TF-IDF artifact {logical_name} is missing") from exc
    if resolved != requested or requested.is_symlink():
        raise ValueError(f"TF-IDF artifact {logical_name} traverses a symlink")
    file_stat = requested.lstat()
    if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
        raise ValueError(f"TF-IDF artifact {logical_name} must be one unlinked regular file")
    digest = hashlib.sha256()
    size = 0
    with requested.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {
        "logical_name": logical_name,
        "size_bytes": int(size),
        "sha256": digest.hexdigest(),
    }


def tfidf_context_artifact_inventory(artifacts: Mapping[str, Any]) -> Dict[str, Any]:
    """Authenticate the complete native context registry without path identity."""

    if set(artifacts) != {
        "fitted_context",
        "ngram_scores",
        "fit_topic_values",
        "heldout_topic_values",
        "nuisance_predictions",
        "topic_score_tests",
    }:
        raise ValueError("TF-IDF native artifact registry is not closed")
    ngram_scores = artifacts.get("ngram_scores")
    if not isinstance(ngram_scores, Mapping) or set(ngram_scores) != {
        "treatment",
        "outcome",
        "effect",
    }:
        raise ValueError("TF-IDF native n-gram artifact registry is not closed")
    entries: List[Dict[str, Any]] = []
    for logical_name in ("fitted_context", "fit_topic_values", "heldout_topic_values"):
        path = artifacts.get(logical_name)
        entry = _registered_regular_file(path, logical_name=logical_name)
        entry.update(
            {
                "artifact_layout": "closed_json_and_per_array_npy_v1",
                "transitive_content_sha256": safe_artifact_content_sha256(path),
            }
        )
        entries.append(entry)
    for bank in ("treatment", "outcome", "effect"):
        entries.append(
            _registered_regular_file(
                ngram_scores[bank],
                logical_name=f"ngram_scores.{bank}",
            )
        )
    entries.append(
        _registered_regular_file(
            artifacts["nuisance_predictions"],
            logical_name="nuisance_predictions",
        )
    )
    if artifacts.get("topic_score_tests") is not None:
        entries.append(
            _registered_regular_file(
                artifacts["topic_score_tests"],
                logical_name="topic_score_tests",
            )
        )
    body = {
        "schema_version": TFIDF_CONTEXT_ARTIFACT_INVENTORY_SCHEMA,
        "entries": entries,
        "topic_score_tests_present": artifacts.get("topic_score_tests") is not None,
    }
    return {**body, "content_sha256": stable_hash(body)}


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


def _fit_base_predict(
    *,
    fit_texts: Sequence[str],
    fit_values: np.ndarray,
    predict_texts: Sequence[str],
    view: BoWViewConfig,
    binary: bool,
    seed: int,
) -> Tuple[np.ndarray, Optional[TfidfVectorizer], Optional[Any]]:
    if binary and len(np.unique(fit_values.astype(int))) < 2:
        return _constant_prediction(fit_values, len(predict_texts)), None, None
    vectorizer = _vectorizer(view)
    try:
        x_fit = vectorizer.fit_transform(fit_texts)
    except ValueError as exc:
        if view.empty_vocabulary_policy == "fail_closed":
            raise ValueError(
                f"BoW view {view.name!r} could not fit its configured vocabulary"
            ) from exc
        return _constant_prediction(fit_values, len(predict_texts)), None, None
    model = _classifier(view, seed) if binary else _regressor(view, seed)
    model.fit(x_fit, fit_values.astype(int) if binary else fit_values)
    if len(predict_texts) == 0:
        return np.zeros(0, dtype=float), vectorizer, model
    x_predict = vectorizer.transform(predict_texts)
    if binary:
        predictions = model.predict_proba(x_predict)[:, 1]
    else:
        predictions = model.predict(x_predict)
    return np.asarray(predictions, dtype=float), vectorizer, model


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
        multi_class=str(config.multi_class),
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


def fit_cross_fitted_nuisance_stack(
    *,
    texts: Sequence[str],
    values: Sequence[float],
    views: Sequence[BoWViewConfig],
    folds: int,
    binary: bool,
    random_state: int,
    nuisance_stack_config: TfidfNuisanceStackScientificConfig,
) -> Dict[str, Any]:
    """Fit a two-level honest OOF stack within one analysis context.

    Each top-level held-out nuisance fold receives base predictions from models
    fitted without that fold.  The stack for that fold is itself trained on
    sub-inner OOF predictions from only the top-level fitting portion.
    """
    if type(nuisance_stack_config) is not TfidfNuisanceStackScientificConfig:
        raise TypeError("nuisance_stack_config must be explicitly typed")
    texts = _normalize_texts(texts)
    values = np.asarray(values, dtype=float)
    views = list(views)
    n_rows = len(values)
    fold_count = _bounded_folds(folds, values, stratified=binary)
    base_oof = np.full((n_rows, len(views)), np.nan, dtype=float)
    stack_oof = np.full(n_rows, np.nan, dtype=float)
    fold_ids = np.full(n_rows, -1, dtype=int)
    fit_ids_by_row: List[List[int]] = [[] for _ in range(n_rows)]

    for fold, (fit_pos, heldout_pos) in enumerate(
        _splitter(
            values,
            fold_count,
            stratified=binary,
            seed=random_state,
            shuffle=nuisance_stack_config.split_shuffle,
        ),
        start=1,
    ):
        fit_pos = np.asarray(fit_pos, dtype=int)
        heldout_pos = np.asarray(heldout_pos, dtype=int)
        sub_values = values[fit_pos]
        sub_folds = _bounded_folds(
            min(fold_count, len(fit_pos) // 2), sub_values, stratified=binary
        )
        meta_fit = np.full((len(fit_pos), len(views)), np.nan, dtype=float)
        for view_index, view in enumerate(views):
            for sub_fold, (sub_fit_local, sub_hold_local) in enumerate(
                _splitter(
                    sub_values,
                    sub_folds,
                    stratified=binary,
                    seed=random_state + 1000 * fold + view_index,
                    shuffle=nuisance_stack_config.split_shuffle,
                ),
                start=1,
            ):
                sub_fit_local = np.asarray(sub_fit_local, dtype=int)
                sub_hold_local = np.asarray(sub_hold_local, dtype=int)
                prediction, _, _ = _fit_base_predict(
                    fit_texts=[texts[fit_pos[index]] for index in sub_fit_local],
                    fit_values=sub_values[sub_fit_local],
                    predict_texts=[texts[fit_pos[index]] for index in sub_hold_local],
                    view=view,
                    binary=binary,
                    seed=random_state + 10_000 * fold + 100 * view_index + sub_fold,
                )
                meta_fit[sub_hold_local, view_index] = prediction
            held_prediction, _, _ = _fit_base_predict(
                fit_texts=[texts[index] for index in fit_pos],
                fit_values=values[fit_pos],
                predict_texts=[texts[index] for index in heldout_pos],
                view=view,
                binary=binary,
                seed=random_state + 20_000 * fold + view_index,
            )
            base_oof[heldout_pos, view_index] = held_prediction
        if not np.isfinite(meta_fit).all():
            raise RuntimeError("Nested nuisance meta-features are incomplete")
        stack_model, stack_constant = _fit_stack(
            meta_fit,
            sub_values,
            binary=binary,
            seed=random_state + 30_000 + fold,
            config=nuisance_stack_config,
        )
        held_meta = base_oof[heldout_pos]
        if stack_model is None:
            stack_oof[heldout_pos] = stack_constant
        elif binary:
            stack_oof[heldout_pos] = stack_model.predict_proba(held_meta)[:, 1]
        else:
            stack_oof[heldout_pos] = stack_model.predict(held_meta)
        fold_ids[heldout_pos] = fold
        fit_list = fit_pos.astype(int).tolist()
        for row in heldout_pos:
            fit_ids_by_row[int(row)] = fit_list

    if not np.isfinite(base_oof).all() or not np.isfinite(stack_oof).all():
        raise RuntimeError("Cross-fitted nuisance predictions are incomplete")

    complete_stack, complete_constant = _fit_stack(
        base_oof,
        values,
        binary=binary,
        seed=random_state + 40_000,
        config=nuisance_stack_config,
    )
    complete_bases: List[Tuple[Optional[TfidfVectorizer], Optional[Any], float]] = []
    for view_index, view in enumerate(views):
        _, vectorizer, model = _fit_base_predict(
            fit_texts=texts,
            fit_values=values,
            predict_texts=[],
            view=view,
            binary=binary,
            seed=random_state + 50_000 + view_index,
        )
        complete_bases.append((vectorizer, model, float(np.mean(values))))
    fitted = CrossFittedStack(
        views=views,
        binary=binary,
        base_models=complete_bases,
        stack_model=complete_stack,
        stack_constant=complete_constant,
        config_hash=stable_hash(
            {
                "views": [asdict(view) for view in views],
                "folds": fold_count,
                "binary": binary,
                "random_state": random_state,
                "nuisance_stack_scientific": asdict(nuisance_stack_config),
            }
        ),
    )
    return {
        "base_oof": base_oof,
        "stacked_oof": stack_oof,
        "fold_ids": fold_ids,
        "fit_positions_by_row": fit_ids_by_row,
        "fitted": fitted,
        "metrics": nuisance_metrics(
            values,
            stack_oof,
            binary=binary,
            calibration_config=nuisance_stack_config.calibration,
        ),
        "view_metrics": {
            view.name: nuisance_metrics(
                values,
                base_oof[:, index],
                binary=binary,
                calibration_config=nuisance_stack_config.calibration,
            )
            for index, view in enumerate(views)
        },
    }


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
    worker_pid: int
    started_monotonic_ns: int
    finished_monotonic_ns: int
    subfold_parallelism: int


def _resolve_joint_nuisance_parallelism(
    *,
    tfidf_workers: int,
    tfidf_parallel_backend: str,
    owner_cpu_budget: Optional[int],
    task_count: int,
) -> Tuple[int, str, str, int]:
    """Validate deployment controls and return the bounded Joblib plan."""

    if isinstance(tfidf_workers, bool) or not isinstance(tfidf_workers, int):
        raise TypeError("tfidf_workers must be a positive integer")
    configured_workers = int(tfidf_workers)
    if configured_workers < 1:
        raise ValueError("tfidf_workers must be a positive integer")
    if owner_cpu_budget is None:
        budget = configured_workers
    else:
        if isinstance(owner_cpu_budget, bool) or not isinstance(owner_cpu_budget, int):
            raise TypeError("TF-IDF owner_cpu_budget must be a positive integer")
        budget = int(owner_cpu_budget)
        if budget < 1:
            raise ValueError("TF-IDF owner_cpu_budget must be a positive integer")
    if isinstance(task_count, bool) or int(task_count) < 1:
        raise ValueError("TF-IDF nuisance top-level fold count must be positive")
    configured_backend = str(tfidf_parallel_backend).strip().lower()
    if configured_backend == "loky":
        configured_backend = "processes"
    if configured_backend not in {"threads", "processes"}:
        raise ValueError(
            "tfidf_parallel_backend must be 'threads', 'processes', or 'loky'"
        )
    return (
        min(configured_workers, budget, int(task_count)),
        configured_backend,
        "threading" if configured_backend == "threads" else "loky",
        budget,
    )


def _fit_joint_nuisance_top_level_fold(
    task: _JointNuisanceTopLevelFoldTask,
) -> _JointNuisanceTopLevelFoldResult:
    """Fit one outer nuisance fold; its subfold loop is deliberately serial."""

    if not isinstance(task, _JointNuisanceTopLevelFoldTask):
        raise TypeError("joint nuisance worker requires its typed fold task")
    started_monotonic_ns = time.monotonic_ns()
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
        worker_pid=int(os.getpid()),
        started_monotonic_ns=int(started_monotonic_ns),
        finished_monotonic_ns=int(time.monotonic_ns()),
        subfold_parallelism=1,
    )


def _run_joint_nuisance_top_level_folds(
    *,
    tasks: Sequence[_JointNuisanceTopLevelFoldTask],
    tfidf_workers: int,
    tfidf_parallel_backend: str,
    owner_cpu_budget: Optional[int],
) -> Tuple[
    List[_JointNuisanceTopLevelFoldResult],
    Dict[str, Any],
]:
    """Run independent outer folds and return a strictly canonical merge set."""

    task_list = list(tasks)
    effective_workers, configured_backend, joblib_backend, budget = (
        _resolve_joint_nuisance_parallelism(
            tfidf_workers=tfidf_workers,
            tfidf_parallel_backend=tfidf_parallel_backend,
            owner_cpu_budget=owner_cpu_budget,
            task_count=len(task_list),
        )
    )
    expected_by_fold: Dict[int, _JointNuisanceTopLevelFoldTask] = {}
    for task in task_list:
        fold = int(task.fold)
        if fold in expected_by_fold:
            raise ValueError("joint nuisance task list contains a duplicate fold")
        expected_by_fold[fold] = task
    expected_folds = list(range(1, len(task_list) + 1))
    if list(expected_by_fold) != expected_folds:
        raise ValueError("joint nuisance task list is not in canonical fold order")

    with threadpool_limits(limits=1):
        if effective_workers == 1:
            completion_order_results = [
                _fit_joint_nuisance_top_level_fold(task)
                for task in task_list
            ]
        else:
            parallel_settings: Dict[str, Any] = {
                "backend": joblib_backend,
                "n_jobs": effective_workers,
            }
            if joblib_backend == "loky":
                parallel_settings["inner_max_num_threads"] = 1
            with parallel_config(**parallel_settings):
                completion_order_results = Parallel(
                    batch_size=1,
                    pre_dispatch="all",
                )(
                    delayed(_fit_joint_nuisance_top_level_fold)(task)
                    for task in task_list
                )

    completion_order = [
        int(result.fold)
        for result in sorted(
            completion_order_results,
            key=lambda result: (
                int(result.finished_monotonic_ns),
                int(result.fold),
            ),
        )
    ]
    ordered = sorted(
        completion_order_results,
        key=lambda result: int(result.fold),
    )
    if [int(result.fold) for result in ordered] != expected_folds:
        raise RuntimeError(
            "parallel TF-IDF nuisance execution changed canonical fold coverage"
        )
    for result in ordered:
        expected = expected_by_fold[int(result.fold)]
        if (
            not np.array_equal(
                np.asarray(result.fit_positions, dtype=int),
                np.asarray(expected.fit_positions, dtype=int),
            )
            or not np.array_equal(
                np.asarray(result.heldout_positions, dtype=int),
                np.asarray(expected.heldout_positions, dtype=int),
            )
        ):
            raise RuntimeError(
                "parallel TF-IDF nuisance worker substituted fold positions"
            )
        expected_targets = (
            {"treatment", "outcome", "mu0", "mu1"}
            if expected.outcome_binary
            else {"treatment", "outcome"}
        )
        if (
            set(result.base_predictions) != expected_targets
            or set(result.stacked_predictions) != expected_targets
            or int(result.subfold_parallelism) != 1
            or int(result.finished_monotonic_ns) <= int(result.started_monotonic_ns)
        ):
            raise RuntimeError(
                "parallel TF-IDF nuisance worker returned an invalid fold result"
            )
        for target in expected_targets:
            base = np.asarray(result.base_predictions[target], dtype=float)
            stacked = np.asarray(
                result.stacked_predictions[target],
                dtype=float,
            )
            if (
                base.shape
                != (len(result.heldout_positions), len(expected.views))
                or stacked.shape != (len(result.heldout_positions),)
                or not np.isfinite(base).all()
                or not np.isfinite(stacked).all()
            ):
                raise RuntimeError(
                    "parallel TF-IDF nuisance worker returned misaligned predictions"
                )

    concurrency = 0
    peak_concurrency = 0
    interval_events = sorted(
        [
            event
            for result in ordered
            for event in (
                (
                    int(result.started_monotonic_ns),
                    1,
                ),
                (
                    int(result.finished_monotonic_ns),
                    -1,
                ),
            )
        ],
        # Treat a finish as releasing its worker before a same-tick start.
        key=lambda event: (event[0], event[1]),
    )
    for _timestamp, delta in interval_events:
        concurrency += int(delta)
        peak_concurrency = max(peak_concurrency, concurrency)
    if concurrency != 0 or peak_concurrency > effective_workers:
        raise RuntimeError(
            "parallel TF-IDF nuisance execution exceeded its worker plan"
        )
    worker_pids = sorted(
        {int(result.worker_pid) for result in ordered}
    )
    if effective_workers > 1 and peak_concurrency < 2:
        raise RuntimeError(
            "parallel TF-IDF nuisance execution silently serialized its "
            "top-level folds"
        )
    if (
        effective_workers > 1
        and configured_backend == "processes"
        and len(worker_pids) < 2
    ):
        raise RuntimeError(
            "process-parallel TF-IDF nuisance execution did not use "
            "isolated workers"
        )
    execution = {
        "configured_workers": int(tfidf_workers),
        "effective_workers": int(effective_workers),
        "actual_peak_concurrent_fold_workers": int(peak_concurrency),
        "owner_cpu_budget": int(budget),
        "worker_cpu_threads": 1,
        "peak_active_worker_cpu_threads": int(peak_concurrency),
        "configured_backend": configured_backend,
        "joblib_backend": joblib_backend,
        "fold_count": len(task_list),
        "canonical_fold_order": expected_folds,
        "completion_order": completion_order,
        "fold_overlap_observed": bool(peak_concurrency > 1),
        "worker_pids": worker_pids,
        "subfold_parallelism": 1,
        "subfold_joblib_pools_created": False,
        "clock_domain_id": "python_monotonic_ns_systemwide_v1",
        "fold_intervals": [
            {
                "fold": int(result.fold),
                "worker_pid": int(result.worker_pid),
                "started_monotonic_ns": int(result.started_monotonic_ns),
                "finished_monotonic_ns": int(result.finished_monotonic_ns),
            }
            for result in ordered
        ],
    }
    return ordered, execution


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
    tfidf_parallel_backend: str = "processes",
    owner_cpu_budget: Optional[int] = None,
    operational_attestation_sink: Optional[
        Callable[[Mapping[str, Any]], None]
    ] = None,
) -> Dict[str, Dict[str, Any]]:
    """Fit treatment/outcome stacks while sharing every label-free TF-IDF fit."""
    if type(nuisance_stack_config) is not TfidfNuisanceStackScientificConfig:
        raise TypeError("nuisance_stack_config must be explicitly typed")
    if operational_attestation_sink is not None and not callable(
        operational_attestation_sink
    ):
        raise TypeError("operational_attestation_sink must be callable")
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
    fold_results, execution_attestation = (
        _run_joint_nuisance_top_level_folds(
            tasks=tasks,
            tfidf_workers=tfidf_workers,
            tfidf_parallel_backend=tfidf_parallel_backend,
            owner_cpu_budget=owner_cpu_budget,
        )
    )
    for result in fold_results:
        fold = int(result.fold)
        fit_pos = np.asarray(result.fit_positions, dtype=int)
        heldout_pos = np.asarray(result.heldout_positions, dtype=int)
        if np.any(fold_ids[heldout_pos] != -1):
            raise RuntimeError(
                "parallel TF-IDF nuisance folds overlap held-out positions"
            )
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
        raise RuntimeError(
            "parallel TF-IDF nuisance merge omitted held-out positions"
        )

    full_fit_started_monotonic_ns = time.monotonic_ns()
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
    full_fit_finished_monotonic_ns = time.monotonic_ns()
    latest_fold_finish = max(
        int(row["finished_monotonic_ns"])
        for row in execution_attestation["fold_intervals"]
    )
    if full_fit_started_monotonic_ns < latest_fold_finish:
        raise RuntimeError(
            "TF-IDF full-data nuisance fits started before the fold barrier"
        )
    attestation_body = {
        "schema_version": TFIDF_NUISANCE_EXECUTION_ATTESTATION_SCHEMA,
        **execution_attestation,
        "fold_barrier_monotonic_ns": int(full_fit_started_monotonic_ns),
        "full_fit_started_monotonic_ns": int(
            full_fit_started_monotonic_ns
        ),
        "full_fit_finished_monotonic_ns": int(
            full_fit_finished_monotonic_ns
        ),
        "full_data_base_fits_after_fold_barrier": True,
        "final_stack_fits_after_fold_barrier": True,
        "scientific_outputs_include_operational_metadata": False,
    }
    attestation = {
        **attestation_body,
        "content_sha256": stable_hash(attestation_body),
    }
    if operational_attestation_sink is not None:
        operational_attestation_sink(attestation)
    return output


def legacy_tfidf_nuisance_stack_v1() -> TfidfNuisanceStackScientificConfig:
    """Historical stack/constant behavior for explicitly legacy callers only."""

    return TfidfNuisanceStackScientificConfig(
        empty_vocabulary_policy="empirical_mean_constant",
    )


def legacy_tfidf_views_with_screening_v1(
    views: Sequence[BoWViewConfig],
    *,
    config: TfidfTopicDiscoveryConfig,
) -> tuple[BoWViewConfig, ...]:
    """Add the historical implicit screen view for named legacy callers only."""

    copied = tuple(views)
    source = config.screening_scientific.model_source_view_name
    if any(view.name == source for view in copied):
        return copied
    return (
        *copied,
        BoWViewConfig(
            name=source,
            max_features=config.max_features,
            min_df=config.min_df,
            max_df=config.max_df,
            ngram_range_min=1,
            ngram_range_max=3,
            sublinear_tf=config.sublinear_tf,
            bow_model="linear",
            empty_vocabulary_policy="empirical_mean_constant",
            unsupported_sample_weight_policy=(
                "unweighted_legacy_compatibility"
            ),
        ),
    )


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
    if config.ece_strategy != "uniform_width":
        raise ValueError("unsupported ECE strategy")
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


def legacy_unsigned_linear_screen_v1(
    matrix: Any,
    feature_names: Sequence[str],
    values: Sequence[float],
    *,
    binary: bool,
    logistic_c: float = 1.0,
    ridge_alpha: float = 10.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """Historical direct screen for explicitly non-production callers."""

    view = BoWViewConfig(
        name="legacy_unsigned_linear_screen_v1",
        bow_model="linear",
        logistic_c=float(logistic_c),
        logistic_max_iter=1000,
        ridge_alpha=float(ridge_alpha),
        empty_vocabulary_policy="empirical_mean_constant",
        unsupported_sample_weight_policy="unweighted_legacy_compatibility",
    )
    return unsigned_linear_screen(
        matrix,
        feature_names,
        values,
        binary=binary,
        view=view,
        random_state=int(random_state),
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
    models: List[NMF]
    component_norms: List[np.ndarray]
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
    ) -> Tuple[Optional["ConsensusNMFTopicBank"], np.ndarray]:
        x = sparse.csr_matrix(matrix, dtype=np.float64)
        names = np.asarray(feature_names, dtype=str)
        score_by_name = scores.set_index("feature")
        eligible = scores
        if "eligible" in eligible.columns:
            eligible = eligible[eligible["eligible"].astype(bool)]
        if eligible.empty or x.shape[1] == 0 or x.shape[0] < 2:
            return None, np.zeros((x.shape[0], 0), dtype=float)
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
            return None, np.zeros((x.shape[0], 0), dtype=float)
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
            return None, np.zeros((x.shape[0], 0), dtype=float)
        reason = None
        if actual < config.topic_count:
            reason = (
                f"requested={config.topic_count}; rows={x.shape[0]}; "
                f"selected_terms={len(selected_indices)}"
            )
        models: List[NMF] = []
        normalized_h: List[np.ndarray] = []
        scaled_w: List[np.ndarray] = []
        norms: List[np.ndarray] = []
        for seed in config.topic_seeds:
            model = NMF(
                n_components=actual,
                init=config.nmf_init,
                solver=config.nmf_solver,
                beta_loss=config.nmf_beta_loss,
                max_iter=config.nmf_max_iter,
                tol=config.nmf_tol,
                random_state=int(seed),
                alpha_W=float(config.nmf_scientific.alpha_w),
                alpha_H=config.nmf_scientific.alpha_h,
                l1_ratio=float(config.nmf_scientific.l1_ratio),
                verbose=0,
                shuffle=bool(config.nmf_scientific.shuffle),
            )
            w = fit_transform_nmf_with_c_order_initial_w(model, weighted)
            h = np.asarray(model.components_, dtype=float)
            component_norm = np.linalg.norm(h, axis=1)
            component_norm = np.where(component_norm > 0.0, component_norm, 1.0)
            models.append(model)
            normalized_h.append(h / component_norm[:, None])
            scaled_w.append(w * component_norm[None, :])
            norms.append(component_norm)
        reference = normalized_h[0]
        alignments: List[np.ndarray] = [np.arange(actual, dtype=int)]
        aligned_h = [reference]
        aligned_w = [scaled_w[0]]
        for index in range(1, len(models)):
            permutation = align_topic_components(reference, normalized_h[index])
            alignments.append(permutation)
            aligned_h.append(normalized_h[index][permutation])
            aligned_w.append(scaled_w[index][:, permutation])
        consensus_h = np.mean(np.stack(aligned_h), axis=0)
        consensus_w = np.mean(np.stack(aligned_w), axis=0)

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
            models=models,
            component_norms=norms,
            alignments=alignments,
            consensus_loadings=consensus_h,
            topic_terms=terms,
            requested_components=int(config.topic_count),
            actual_components=int(actual),
            terms_per_topic=int(config.terms_per_topic),
            seeds=list(config.topic_seeds),
            reduction_reason=reason,
        )
        return bank, consensus_w

    def transform(self, matrix: Any) -> np.ndarray:
        x = sparse.csr_matrix(matrix, dtype=np.float64)
        weighted = x[:, self.selected_indices].multiply(self.feature_weights).tocsr()
        aligned = []
        for model, norms, permutation in zip(self.models, self.component_norms, self.alignments):
            values = model.transform(weighted) * norms[None, :]
            aligned.append(values[:, permutation])
        return np.mean(np.stack(aligned), axis=0)

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


@dataclass
class FittedTopicContext:
    common_vectorizer: TfidfVectorizer
    treatment_stack: CrossFittedStack
    outcome_stack: CrossFittedStack
    topic_banks: Dict[str, ConsensusNMFTopicBank]
    config_hash: str

    def transform_topics(self, texts: Sequence[str]) -> Dict[str, np.ndarray]:
        matrix = self.common_vectorizer.transform(_normalize_texts(texts))
        return {name: bank.transform(matrix) for name, bank in self.topic_banks.items()}


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


def compact_topic_score_tests(score_tests: Mapping[str, Any]) -> Dict[str, Any]:
    """Build the small handoff view while retaining auditable score evidence."""
    compact: Dict[str, Any] = {
        "schema_version": score_tests.get("schema_version", TOPIC_SCORE_TEST_SCHEMA_VERSION),
        "status": score_tests.get("status", "not_run"),
        "reason": score_tests.get("reason"),
        "terms_per_topic": score_tests.get("terms_per_topic"),
        "uses_heldout_treatment_and_outcome": bool(
            score_tests.get("uses_heldout_treatment_and_outcome", False)
        ),
        "uses_registered_heldout_treatment_and_outcome": bool(
            score_tests.get("uses_registered_heldout_treatment_and_outcome", False)
        ),
        "uses_nested_fit_calibration_treatment_and_outcome": bool(
            score_tests.get("uses_nested_fit_calibration_treatment_and_outcome", False)
        ),
        "score_selection_label_policy": score_tests.get(
            "score_selection_label_policy", "registered_context_heldout"
        ),
        "selection_frozen_sha256": score_tests.get("selection_frozen_sha256"),
        "banks": {},
    }
    for bank_name, bank_result in (score_tests.get("banks") or {}).items():
        compact["banks"][bank_name] = {
            key: bank_result.get(key)
            for key in (
                "selected_topic_ids",
                "selection_count",
                "selection_rule",
                "fdr_level",
                "p_threshold",
                "minimum_topics",
                "maximum_topics",
                "complete_family_multiplier_bootstrap",
                "complete_term_group_multiplier_bootstrap",
                "selected_ngram_terms",
                "ngram_selection_count",
                "ngram_selection_rule",
                "unique_ngram_count",
                "testable_unique_ngram_count",
            )
        }
        compact["banks"][bank_name]["bootstrap_calibration"] = (
            bank_result.get("bootstrap_calibration") or {}
        )
        compact["banks"][bank_name]["topic_ranking"] = [
            {
                key: topic_result.get(key)
                for key in (
                    "topic_id",
                    "evidence_rank",
                    "selected_for_agent",
                    "selection_reason",
                    "primary_p",
                    "primary_p_source",
                    "fdr_q",
                    "familywise_p",
                    "topic_score_testable",
                    "topic_score_moment",
                    "topic_standardized_score",
                    "topic_unadjusted_two_sided_p",
                    "topic_familywise_p",
                    "term_group_primary_p",
                    "term_group_primary_p_source",
                    "term_group_fdr_q",
                    "term_group_familywise_p",
                    "quadratic_statistic_per_rank",
                    "maximum_absolute_standardized_score",
                    "selected_ngram_count",
                    "selected_ngram_terms",
                )
            }
            for topic_result in bank_result.get("topic_tests", [])
        ]
    orphan = score_tests.get("effect_orphan_ngram_branch") or {}
    compact["effect_orphan_ngram_branch"] = {
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
    return compact


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
    enable_heldout_score_tests: bool = False,
    tfidf_workers: int = 1,
    tfidf_parallel_backend: str = "processes",
    owner_cpu_budget: Optional[int] = None,
    operational_attestation_sink: Optional[
        Callable[[Mapping[str, Any]], None]
    ] = None,
) -> Dict[str, Any]:
    """Fit one exact context and transform its externally held-out row set."""
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
        tfidf_workers=tfidf_workers,
        tfidf_parallel_backend=tfidf_parallel_backend,
        owner_cpu_budget=owner_cpu_budget,
        operational_attestation_sink=operational_attestation_sink,
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

    vectorizer = _common_vectorizer(config)
    common_fit = vectorizer.fit_transform(fit_texts)
    common_heldout = vectorizer.transform(heldout_texts)
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
    topic_banks: Dict[str, ConsensusNMFTopicBank] = {}
    fit_topic_values: Dict[str, np.ndarray] = {}
    heldout_topic_values: Dict[str, np.ndarray] = {}
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
        bank, fit_values = ConsensusNMFTopicBank.fit(
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
        topic_banks[bank_name] = bank
        fit_topic_values[bank_name] = fit_values
        heldout_topic_values[bank_name] = bank.transform(common_heldout)
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

    topic_score_tests: Dict[str, Any] = {
        "schema_version": TOPIC_SCORE_TEST_SCHEMA_VERSION,
        "status": "not_run",
        "reason": "heldout_labels_reserved_or_score_tests_disabled",
        "terms_per_topic": int(config.terms_per_topic),
        "uses_heldout_treatment_and_outcome": False,
        "banks": {},
    }
    topic_score_tests_path: Optional[Path] = None
    if bool(config.score_test_enabled) and bool(enable_heldout_score_tests):
        # This block is used only for exact candidate-selection inner contexts.
        # Full outer contexts never read outer-held-out treatment or outcome.
        topic_score_tests = score_topic_banks(
            dina_nuisance=(dn, dv) if dn is not None else None,
            fit_matrix=common_fit,
            heldout_matrix=common_heldout,
            feature_names=feature_names,
            topic_banks=bank_metadata,
            fit_topic_values=fit_topic_values,
            heldout_topic_values=heldout_topic_values,
            fit_treatment=treatment,
            fit_outcome=outcome,
            heldout_treatment=heldout_df[treatment_column].to_numpy(dtype=float),
            heldout_outcome=heldout_df[outcome_column].to_numpy(dtype=float),
            fit_propensity=treatment_result["stacked_oof"],
            fit_outcome_prediction=outcome_result["stacked_oof"],
            heldout_propensity=external_e,
            heldout_outcome_prediction=external_m,
            config=config,
            scope_id=scope_id,
            raw_ngram_scores=score_frames,
        )
        topic_score_tests["status"] = "completed"
        topic_score_tests_path = artifact_dir / "topic_score_tests.json"
        topic_score_tests_path.write_text(
            json.dumps(topic_score_tests, indent=2, default=str),
            encoding="utf-8",
        )

    compact_score_tests = compact_topic_score_tests(topic_score_tests)

    fitted = FittedTopicContext(
        common_vectorizer=vectorizer,
        treatment_stack=treatment_result["fitted"],
        outcome_stack=outcome_result["fitted"],
        topic_banks=topic_banks,
        config_hash=stable_hash(asdict(config)),
    )
    model_path = write_fitted_topic_context(
        fitted,
        artifact_dir / "fitted_context",
    )
    fit_topics_path = write_named_array_bank(
        fit_topic_values,
        artifact_dir / "fit_topic_values",
        row_count=len(fit_df),
    )
    heldout_topics_path = write_named_array_bank(
        heldout_topic_values,
        artifact_dir / "heldout_topic_values",
        row_count=len(heldout_df),
    )

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
        "config_hash": fitted.config_hash,
        "common_vocabulary_size": int(len(feature_names)),
        "common_vocabulary": feature_names.astype(str).tolist(),
        "common_vocabulary_selection": {
            "selected_term_count": int(len(feature_names)),
            "configured_max_features": int(config.max_features),
            "selection_rule": (
                config.vectorizer_scientific.feature_selection_rule
            ),
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
        "heldout_score_tests_enabled": bool(enable_heldout_score_tests),
        "topic_score_tests": compact_score_tests,
        "artifacts": {
            "fitted_context": str(model_path),
            "ngram_scores": score_paths,
            "fit_topic_values": str(fit_topics_path),
            "heldout_topic_values": str(heldout_topics_path),
            "nuisance_predictions": str(nuisance_path),
            "topic_score_tests": (
                None if topic_score_tests_path is None else str(topic_score_tests_path)
            ),
        },
    }
    metadata["artifact_inventory"] = tfidf_context_artifact_inventory(
        metadata["artifacts"]
    )
    metadata_path = artifact_dir / "context_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return metadata
