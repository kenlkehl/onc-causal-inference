"""Matched-pair uplift helpers for the multi-model forest path."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize
from scipy import sparse
from scipy.special import expit, logit
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import KFold

from .sparse_text_modeling import (
    _fit_regressor,
    _make_bow_regressor,
    _make_bow_vectorizer,
    _model_feature_scores,
    _top_feature_rows,
)
from .stage1_checkpointing import Stage1CheckpointStore, row_id_identity

logger = logging.getLogger(__name__)

_EPS = 1e-5


@dataclass
class PairUpliftFitResult:
    train_delta_logit: np.ndarray
    test_delta_logit: np.ndarray
    train_pred_prob: np.ndarray
    test_pred_prob: np.ndarray
    train_n_controls: np.ndarray
    test_n_controls: np.ndarray
    feature_importance: Dict[str, Any]
    evidence_rows: List[Dict[str, Any]]
    attention_rows: List[Dict[str, Any]]
    prediction_frame: pd.DataFrame
    metrics: Dict[str, Any]


def probability_logit(prob: Any) -> np.ndarray:
    p = np.clip(np.asarray(prob, dtype=float), _EPS, 1.0 - _EPS)
    return logit(p)


def _finite_or_none(value: Any) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def _safe_roc(y: np.ndarray, pred: np.ndarray) -> Optional[float]:
    mask = np.isfinite(y) & np.isfinite(pred)
    if int(mask.sum()) < 2 or len(np.unique(y[mask].astype(int))) < 2:
        return None
    try:
        return float(roc_auc_score(y[mask].astype(int), pred[mask]))
    except ValueError:
        return None


def _binary_metrics(y: np.ndarray, pred: np.ndarray) -> Dict[str, Any]:
    mask = np.isfinite(y) & np.isfinite(pred)
    metrics: Dict[str, Any] = {"n_eval": int(mask.sum()), "auroc": _safe_roc(y, pred)}
    if int(mask.sum()) < 1:
        metrics.update({"brier": None, "log_loss": None})
        return metrics
    p = np.clip(pred[mask], _EPS, 1.0 - _EPS)
    target = y[mask].astype(int)
    try:
        metrics["brier"] = float(brier_score_loss(target, p))
    except ValueError:
        metrics["brier"] = None
    try:
        metrics["log_loss"] = float(log_loss(target, p, labels=[0, 1]))
    except ValueError:
        metrics["log_loss"] = None
    return metrics


def _bounded_folds(requested: int, n_rows: int) -> int:
    return max(2, min(int(requested), int(n_rows)))


def hopcroft_karp(adjacency: Sequence[Sequence[int]], n_right: int) -> Tuple[List[int], List[int]]:
    n_left = len(adjacency)
    pair_left = [-1] * n_left
    pair_right = [-1] * n_right
    dist = [0] * n_left

    def bfs() -> bool:
        queue: deque[int] = deque()
        found = False
        for left in range(n_left):
            if pair_left[left] == -1:
                dist[left] = 0
                queue.append(left)
            else:
                dist[left] = -1
        while queue:
            left = queue.popleft()
            for right in adjacency[left]:
                matched_left = pair_right[right]
                if matched_left == -1:
                    found = True
                elif dist[matched_left] == -1:
                    dist[matched_left] = dist[left] + 1
                    queue.append(matched_left)
        return found

    def dfs(left: int) -> bool:
        for right in adjacency[left]:
            matched_left = pair_right[right]
            if matched_left == -1 or (
                dist[matched_left] == dist[left] + 1 and dfs(matched_left)
            ):
                pair_left[left] = right
                pair_right[right] = left
                return True
        dist[left] = -1
        return False

    while bfs():
        for left in range(n_left):
            if pair_left[left] == -1:
                dfs(left)
    return pair_left, pair_right


def _empty_pair_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "candidate_pos",
            "treated_pos",
            "control_pos",
            "candidate_row_id",
            "treated_row_id",
            "control_row_id",
            "treated_text",
            "control_text",
            "label",
            "base_prob",
            "base_logit",
            "propensity_abs_diff",
            "outcome_abs_diff",
            "score_abs_diff_sum",
        ]
    )


def build_training_pairs(
    df: pd.DataFrame,
    *,
    texts: Sequence[str],
    treatment: np.ndarray,
    outcome: np.ndarray,
    propensity: np.ndarray,
    outcome_prob: np.ndarray,
    propensity_caliper: float,
    outcome_caliper: float,
) -> pd.DataFrame:
    treated_pos = np.where(np.asarray(treatment, dtype=int) == 1)[0]
    control_pos = np.where(np.asarray(treatment, dtype=int) == 0)[0]
    if len(treated_pos) == 0 or len(control_pos) == 0:
        return _empty_pair_frame()

    t_prop = propensity[treated_pos]
    t_out = outcome_prob[treated_pos]
    c_prop = propensity[control_pos]
    c_out = outcome_prob[control_pos]
    adjacency: List[List[int]] = []
    for i in range(len(treated_pos)):
        prop_diff = np.abs(c_prop - t_prop[i])
        out_diff = np.abs(c_out - t_out[i])
        eligible = np.where(
            (prop_diff <= float(propensity_caliper))
            & (out_diff <= float(outcome_caliper))
        )[0]
        adjacency.append(
            sorted(
                eligible.tolist(),
                key=lambda j: (float(prop_diff[j] + out_diff[j]), float(prop_diff[j]), int(j)),
            )
        )
    pair_left, _ = hopcroft_karp(adjacency, len(control_pos))
    rows = []
    row_ids = df["_oci_row_id"].to_numpy() if "_oci_row_id" in df.columns else np.arange(len(df))
    for left_idx, right_idx in enumerate(pair_left):
        if right_idx < 0:
            continue
        tp = int(treated_pos[left_idx])
        cp = int(control_pos[right_idx])
        prop_diff = float(abs(propensity[tp] - propensity[cp]))
        out_diff = float(abs(outcome_prob[tp] - outcome_prob[cp]))
        rows.append(
            {
                "candidate_pos": tp,
                "treated_pos": tp,
                "control_pos": cp,
                "candidate_row_id": int(row_ids[tp]),
                "treated_row_id": int(row_ids[tp]),
                "control_row_id": int(row_ids[cp]),
                "treated_text": str(texts[tp]),
                "control_text": str(texts[cp]),
                "label": int(outcome[tp]),
                "base_prob": float(np.clip(outcome_prob[cp], _EPS, 1.0 - _EPS)),
                "base_logit": float(probability_logit([outcome_prob[cp]])[0]),
                "propensity_abs_diff": prop_diff,
                "outcome_abs_diff": out_diff,
                "score_abs_diff_sum": prop_diff + out_diff,
            }
        )
    if not rows:
        return _empty_pair_frame()
    return pd.DataFrame(rows)


def build_candidate_pairs(
    candidate_df: pd.DataFrame,
    control_df: pd.DataFrame,
    *,
    candidate_texts: Sequence[str],
    control_texts: Sequence[str],
    candidate_propensity: np.ndarray,
    candidate_outcome_prob: np.ndarray,
    control_propensity: np.ndarray,
    control_outcome_prob: np.ndarray,
    propensity_caliper: float,
    outcome_caliper: float,
    max_controls_per_candidate: int,
    nearest_fallback_controls: int,
) -> pd.DataFrame:
    if len(candidate_df) == 0 or len(control_df) == 0:
        return _empty_pair_frame()
    candidate_row_ids = (
        candidate_df["_oci_row_id"].to_numpy()
        if "_oci_row_id" in candidate_df.columns
        else np.arange(len(candidate_df))
    )
    control_row_ids = (
        control_df["_oci_row_id"].to_numpy()
        if "_oci_row_id" in control_df.columns
        else np.arange(len(control_df))
    )
    rows = []
    max_controls = max(1, int(max_controls_per_candidate))
    fallback_controls = max(0, int(nearest_fallback_controls))
    c_prop = np.asarray(control_propensity, dtype=float)
    c_out = np.asarray(control_outcome_prob, dtype=float)
    for candidate_pos in range(len(candidate_df)):
        prop_diff = np.abs(c_prop - float(candidate_propensity[candidate_pos]))
        out_diff = np.abs(c_out - float(candidate_outcome_prob[candidate_pos]))
        score = prop_diff + out_diff
        eligible = np.where(
            (prop_diff <= float(propensity_caliper))
            & (out_diff <= float(outcome_caliper))
        )[0]
        used_fallback = False
        if len(eligible) == 0 and fallback_controls > 0:
            eligible = np.argsort(score)[:fallback_controls]
            used_fallback = True
        order = sorted(
            eligible.tolist(),
            key=lambda j: (float(score[j]), float(prop_diff[j]), int(control_row_ids[j])),
        )[:max_controls]
        for control_pos in order:
            rows.append(
                {
                    "candidate_pos": int(candidate_pos),
                    "treated_pos": int(candidate_pos),
                    "control_pos": int(control_pos),
                    "candidate_row_id": int(candidate_row_ids[candidate_pos]),
                    "treated_row_id": int(candidate_row_ids[candidate_pos]),
                    "control_row_id": int(control_row_ids[control_pos]),
                    "treated_text": str(candidate_texts[candidate_pos]),
                    "control_text": str(control_texts[control_pos]),
                    "label": np.nan,
                    "base_prob": float(np.clip(c_out[control_pos], _EPS, 1.0 - _EPS)),
                    "base_logit": float(probability_logit([c_out[control_pos]])[0]),
                    "propensity_abs_diff": float(prop_diff[control_pos]),
                    "outcome_abs_diff": float(out_diff[control_pos]),
                    "score_abs_diff_sum": float(score[control_pos]),
                    "used_nearest_fallback": bool(used_fallback),
                }
            )
    if not rows:
        return _empty_pair_frame()
    return pd.DataFrame(rows)


def aggregate_pair_predictions(
    pairs: pd.DataFrame,
    delta_logit: np.ndarray,
    n_candidates: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    delta = np.full(n_candidates, np.nan, dtype=float)
    pred_prob = np.full(n_candidates, np.nan, dtype=float)
    n_controls = np.zeros(n_candidates, dtype=float)
    if pairs.empty:
        return delta, pred_prob, n_controls
    work = pairs[["candidate_pos", "base_logit"]].copy()
    work["delta_logit"] = np.asarray(delta_logit, dtype=float)
    work["pred_prob"] = expit(work["base_logit"].to_numpy(dtype=float) + work["delta_logit"])
    grouped = work.groupby("candidate_pos", sort=False)
    for candidate_pos, group in grouped:
        pos = int(candidate_pos)
        delta[pos] = float(group["delta_logit"].mean())
        pred_prob[pos] = float(group["pred_prob"].mean())
        n_controls[pos] = float(len(group))
    return delta, pred_prob, n_controls


class OffsetLogitBoWPairModel:
    def __init__(
        self,
        *,
        vectorizer_params: Dict[str, Any],
        l2_alpha: float,
        max_iter: int,
        random_state: int,
        optimizer_method: str = "L-BFGS-B",
        optimizer_ftol: float = 1e-8,
        optimizer_gtol: float = 1e-5,
        optimizer_maxls: int = 30,
        optimizer_maxcor: int = 10,
        optimizer_maxfun: int = 15_000,
        optimizer_tol: Optional[float] = None,
        optimizer_initialization: str = "zeros",
        require_optimizer_success: bool = False,
    ) -> None:
        del random_state
        self.vectorizer_params = dict(vectorizer_params)
        self.l2_alpha = float(l2_alpha)
        self.max_iter = int(max_iter)
        self.optimizer_method = str(optimizer_method)
        self.optimizer_ftol = float(optimizer_ftol)
        self.optimizer_gtol = float(optimizer_gtol)
        self.optimizer_maxls = int(optimizer_maxls)
        self.optimizer_maxcor = int(optimizer_maxcor)
        self.optimizer_maxfun = int(optimizer_maxfun)
        self.optimizer_tol = (
            None if optimizer_tol is None else float(optimizer_tol)
        )
        self.optimizer_initialization = str(optimizer_initialization)
        self.require_optimizer_success = bool(require_optimizer_success)
        if self.optimizer_method != "L-BFGS-B":
            raise ValueError("offset-logit pair model supports only L-BFGS-B")
        if self.optimizer_initialization != "zeros":
            raise ValueError("offset-logit pair model supports only zero initialization")
        if (
            self.max_iter < 1
            or self.optimizer_maxls < 1
            or self.optimizer_maxcor < 1
            or self.optimizer_maxfun < 1
            or not np.isfinite(
                [
                    self.optimizer_ftol,
                    self.optimizer_gtol,
                    0.0 if self.optimizer_tol is None else self.optimizer_tol,
                ]
            ).all()
            or self.optimizer_ftol < 0.0
            or self.optimizer_gtol < 0.0
            or (
                self.optimizer_tol is not None
                and self.optimizer_tol < 0.0
            )
        ):
            raise ValueError("offset-logit L-BFGS-B configuration is invalid")
        self.vectorizer = None
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.constant_delta_: Optional[float] = None

    def _matrix(self, pairs: pd.DataFrame):
        if self.vectorizer is None:
            raise RuntimeError("BoW pair model has not been fitted.")
        control = self.vectorizer.transform(pairs["control_text"].astype(str).tolist())
        treated = self.vectorizer.transform(pairs["treated_text"].astype(str).tolist())
        return sparse.hstack([control, treated], format="csr")

    def fit(self, pairs: pd.DataFrame) -> "OffsetLogitBoWPairModel":
        if pairs.empty:
            self.constant_delta_ = 0.0
            return self
        y = pairs["label"].to_numpy(dtype=float)
        offset = pairs["base_logit"].to_numpy(dtype=float)
        if len(np.unique(y.astype(int))) < 2:
            self.constant_delta_ = float(probability_logit([np.mean(y)])[0] - np.mean(offset))
            return self
        try:
            self.vectorizer = _make_bow_vectorizer(self.vectorizer_params)
            self.vectorizer.fit(
                pairs["control_text"].astype(str).tolist()
                + pairs["treated_text"].astype(str).tolist()
            )
            x = self._matrix(pairs)
        except ValueError:
            self.constant_delta_ = float(probability_logit([np.mean(y)])[0] - np.mean(offset))
            return self

        n_obs, n_features = x.shape

        def objective(params: np.ndarray) -> Tuple[float, np.ndarray]:
            intercept = params[0]
            coef = params[1:]
            eta = offset + intercept + x.dot(coef)
            prob = expit(eta)
            residual = prob - y
            loss = float(np.logaddexp(0.0, eta).mean() - np.mean(y * eta))
            penalty = 0.5 * self.l2_alpha * float(coef @ coef)
            grad_intercept = np.array([residual.mean()])
            grad_coef = np.asarray(x.T.dot(residual)).ravel() / n_obs + self.l2_alpha * coef
            return loss + penalty, np.concatenate([grad_intercept, grad_coef])

        result = minimize(
            objective,
            np.zeros(n_features + 1, dtype=float),
            method=self.optimizer_method,
            jac=True,
            tol=self.optimizer_tol,
            options={
                "maxiter": self.max_iter,
                "ftol": self.optimizer_ftol,
                "gtol": self.optimizer_gtol,
                "maxls": self.optimizer_maxls,
                "maxcor": self.optimizer_maxcor,
                "maxfun": self.optimizer_maxfun,
            },
        )
        if not result.success:
            if self.require_optimizer_success:
                raise RuntimeError(
                    "BoW pair uplift L-BFGS-B did not converge: "
                    f"{result.message}"
                )
            logger.warning("BoW pair uplift optimizer ended with: %s", result.message)
        self.intercept_ = float(result.x[0])
        self.coef_ = np.asarray(result.x[1:], dtype=float)
        return self

    def predict_delta_logit(self, pairs: pd.DataFrame) -> np.ndarray:
        if pairs.empty:
            return np.zeros(0, dtype=float)
        if self.constant_delta_ is not None or self.vectorizer is None or self.coef_ is None:
            return np.full(len(pairs), float(self.constant_delta_ or 0.0), dtype=float)
        return np.asarray(self.intercept_ + self._matrix(pairs).dot(self.coef_), dtype=float)

    def top_features(self, top_n: int) -> Dict[str, Any]:
        if self.vectorizer is None or self.coef_ is None:
            return {
                "n_features": 0,
                "uplift_delta_logit_positive": [],
                "uplift_delta_logit_negative": [],
                "uplift_pair_features": [],
            }
        terms = self.vectorizer.get_feature_names_out().astype(str)
        names = np.concatenate([np.char.add("control::", terms), np.char.add("treated::", terms)])
        coef = np.asarray(self.coef_, dtype=float)
        positive = _top_feature_rows(names, coef, top_n)
        negative = _top_feature_rows(names, coef, top_n, descending=False)
        phrase = _top_feature_rows(names, np.abs(coef), top_n)
        for row in positive + negative + phrase:
            row["uplift_delta_logit_score"] = row.get("score")
            row["abs_uplift_delta_logit_score"] = _finite_or_none(abs(row.get("score") or 0.0))
        return {
            "n_features": int(len(names)),
            "uplift_delta_logit_positive": positive,
            "uplift_delta_logit_negative": negative,
            "uplift_pair_features": phrase,
        }


class RidgeDeltaBoWPairModel:
    """Optional probability-scale companion used for diagnostics."""

    def __init__(
        self,
        *,
        vectorizer_params: Dict[str, Any],
        model_params: Dict[str, Any],
        random_state: int,
    ) -> None:
        self.vectorizer_params = dict(vectorizer_params)
        self.model_params = dict(model_params)
        self.random_state = int(random_state)
        self.vectorizer = None
        self.model = None

    def _matrix(self, pairs: pd.DataFrame):
        if self.vectorizer is None:
            raise RuntimeError("BoW pair model has not been fitted.")
        control = self.vectorizer.transform(pairs["control_text"].astype(str).tolist())
        treated = self.vectorizer.transform(pairs["treated_text"].astype(str).tolist())
        return sparse.hstack([control, treated], format="csr")

    def fit(self, pairs: pd.DataFrame) -> "RidgeDeltaBoWPairModel":
        self.model = _make_bow_regressor(self.model_params, random_state=self.random_state)
        if pairs.empty:
            return self
        self.vectorizer = _make_bow_vectorizer(self.vectorizer_params)
        self.vectorizer.fit(
            pairs["control_text"].astype(str).tolist()
            + pairs["treated_text"].astype(str).tolist()
        )
        y = pairs["label"].to_numpy(dtype=float) - pairs["base_prob"].to_numpy(dtype=float)
        _fit_regressor(
            self.model,
            self._matrix(pairs),
            y,
            unsupported_sample_weight_policy=str(
                self.model_params["unsupported_sample_weight_policy"]
            ),
        )
        return self

    def predict_delta_prob(self, pairs: pd.DataFrame) -> np.ndarray:
        if pairs.empty:
            return np.zeros(0, dtype=float)
        if self.vectorizer is None or self.model is None:
            return np.zeros(len(pairs), dtype=float)
        return np.asarray(self.model.predict(self._matrix(pairs)), dtype=float)

    def top_features(self, top_n: int) -> Dict[str, Any]:
        if self.vectorizer is None or self.model is None:
            return {"ridge_delta_probability_positive": [], "ridge_delta_probability_negative": []}
        terms = self.vectorizer.get_feature_names_out().astype(str)
        names = np.concatenate([np.char.add("control::", terms), np.char.add("treated::", terms)])
        coef = _model_feature_scores(self.model, len(names))
        return {
            "ridge_delta_probability_positive": _top_feature_rows(names, coef, top_n),
            "ridge_delta_probability_negative": _top_feature_rows(
                names,
                coef,
                top_n,
                descending=False,
            ),
        }


def _fit_bow_pair_uplift_fold(
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
    vectorizer_params: Dict[str, Any],
    outer_fold: int,
    view_name: str,
    view_index: int,
    inner_fold: int,
    fit_pos: np.ndarray,
    heldout_pos: np.ndarray,
    propensity_caliper: float,
    outcome_caliper: float,
    max_controls_per_candidate: int,
    nearest_fallback_controls: int,
    l2_alpha: float,
    max_iter: int,
    native_capture_sink: Optional[Any],
) -> Dict[str, Any]:
    """Fit and score one independently checkpointable BoW pair fold."""

    fit_pos = np.asarray(fit_pos, dtype=int)
    heldout_pos = np.asarray(heldout_pos, dtype=int)
    fit_df = train_df.iloc[fit_pos].reset_index(drop=True)
    heldout_df = train_df.iloc[heldout_pos].reset_index(drop=True)
    fit_pairs = build_training_pairs(
        fit_df,
        texts=[texts_train[int(pos)] for pos in fit_pos],
        treatment=t_train[fit_pos],
        outcome=y_train[fit_pos],
        propensity=e_train[fit_pos],
        outcome_prob=m_train[fit_pos],
        propensity_caliper=propensity_caliper,
        outcome_caliper=outcome_caliper,
    )
    model = OffsetLogitBoWPairModel(
        vectorizer_params=vectorizer_params,
        l2_alpha=l2_alpha,
        max_iter=max_iter,
        random_state=31_000 + int(inner_fold),
    ).fit(fit_pairs)
    control_pos = fit_pos[t_train[fit_pos].astype(int) == 0]
    control_df = train_df.iloc[control_pos].reset_index(drop=True)
    heldout_pairs = build_candidate_pairs(
        heldout_df,
        control_df,
        candidate_texts=[texts_train[int(pos)] for pos in heldout_pos],
        control_texts=[texts_train[int(pos)] for pos in control_pos],
        candidate_propensity=e_train[heldout_pos],
        candidate_outcome_prob=m_train[heldout_pos],
        control_propensity=e_train[control_pos],
        control_outcome_prob=m_train[control_pos],
        propensity_caliper=propensity_caliper,
        outcome_caliper=outcome_caliper,
        max_controls_per_candidate=max_controls_per_candidate,
        nearest_fallback_controls=nearest_fallback_controls,
    )
    heldout_pair_delta = model.predict_delta_logit(heldout_pairs)
    fold_delta, fold_prob, fold_n = aggregate_pair_predictions(
        heldout_pairs,
        heldout_pair_delta,
        len(heldout_df),
    )
    test_pairs = build_candidate_pairs(
        test_df,
        control_df,
        candidate_texts=texts_test,
        control_texts=[texts_train[int(pos)] for pos in control_pos],
        candidate_propensity=e_test,
        candidate_outcome_prob=m_test,
        control_propensity=e_train[control_pos],
        control_outcome_prob=m_train[control_pos],
        propensity_caliper=propensity_caliper,
        outcome_caliper=outcome_caliper,
        max_controls_per_candidate=max_controls_per_candidate,
        nearest_fallback_controls=nearest_fallback_controls,
    )
    test_pair_delta = model.predict_delta_logit(test_pairs)
    fold_test_delta, fold_test_prob, fold_test_n = aggregate_pair_predictions(
        test_pairs,
        test_pair_delta,
        len(test_df),
    )
    if native_capture_sink is not None:
        native_capture_sink.record_bow_pair_fold(
            view_name=view_name,
            view_index=view_index,
            fold=inner_fold,
            fit_pos=fit_pos,
            validation_pos=heldout_pos,
            fit_pairs=fit_pairs,
            validation_pairs=heldout_pairs,
            heldout_pairs=test_pairs,
            model=model,
            validation_pair_delta=heldout_pair_delta,
            validation_delta=fold_delta,
            validation_probability=fold_prob,
            validation_n_controls=fold_n,
            heldout_pair_delta=test_pair_delta,
            heldout_delta=fold_test_delta,
            heldout_probability=fold_test_prob,
            heldout_n_controls=fold_test_n,
        )
    treated_eval = (t_train[heldout_pos].astype(int) == 1) & np.isfinite(fold_prob)
    evidence = {
        "outer_fold": int(outer_fold),
        "inner_fold": int(inner_fold),
        "source_family": "bow_pair_uplift",
        "view_name": str(view_name),
        "objective": "matched_pair_uplift_delta_logit",
        "target_name": "treated_observed_outcome",
        "train_rows": int(len(fit_pos)),
        "heldout_rows": int(len(heldout_pos)),
        "outer_test_rows": int(len(test_df)),
        "matched_pair_train_rows": int(len(fit_pairs)),
        "heldout_candidate_pair_rows": int(len(heldout_pairs)),
        "outer_test_candidate_pair_rows": int(len(test_pairs)),
        "heldout_treated_auroc": _safe_roc(
            y_train[heldout_pos][treated_eval],
            fold_prob[treated_eval],
        ),
        "prediction_provenance": "inner_fold_pair_model_heldout_and_outer_test",
    }
    prediction_frame = pd.DataFrame()
    if not heldout_pairs.empty:
        prediction_frame = heldout_pairs[["candidate_row_id", "control_row_id"]].copy()
        prediction_frame["outer_fold"] = int(outer_fold)
        prediction_frame["inner_fold"] = int(inner_fold)
        prediction_frame["split_role"] = "train_inner_oof_pairs"
        prediction_frame["source_name"] = f"bow__{view_name}__pair_uplift"
        prediction_frame["pair_delta_logit"] = heldout_pair_delta
        prediction_frame["pair_pred_prob"] = expit(
            heldout_pairs["base_logit"].to_numpy(dtype=float) + heldout_pair_delta
        )
    return {
        "heldout_pos": heldout_pos,
        "fold_delta": fold_delta,
        "fold_prob": fold_prob,
        "fold_n": fold_n,
        "test_delta": fold_test_delta,
        "test_prob": fold_test_prob,
        "test_n": fold_test_n,
        "attention_rows": [],
        "evidence": evidence,
        "prediction_frame": prediction_frame,
    }


def _fit_bow_pair_full_importance(
    *,
    train_df: pd.DataFrame,
    texts_train: Sequence[str],
    y_train: np.ndarray,
    t_train: np.ndarray,
    e_train: np.ndarray,
    m_train: np.ndarray,
    vectorizer_params: Dict[str, Any],
    model_params: Dict[str, Any],
    outer_fold: int,
    view_name: str,
    view_index: int,
    propensity_caliper: float,
    outcome_caliper: float,
    max_controls_per_candidate: int,
    nearest_fallback_controls: int,
    l2_alpha: float,
    max_iter: int,
    top_n: int,
    native_capture_sink: Optional[Any],
) -> Dict[str, Any]:
    full_pairs = build_training_pairs(
        train_df,
        texts=texts_train,
        treatment=t_train,
        outcome=y_train,
        propensity=e_train,
        outcome_prob=m_train,
        propensity_caliper=propensity_caliper,
        outcome_caliper=outcome_caliper,
    )
    full_model = OffsetLogitBoWPairModel(
        vectorizer_params=vectorizer_params,
        l2_alpha=l2_alpha,
        max_iter=max_iter,
        random_state=77_000 + int(outer_fold),
    ).fit(full_pairs)
    importance = full_model.top_features(top_n)
    ridge_model: Optional[RidgeDeltaBoWPairModel] = None
    try:
        ridge_model = RidgeDeltaBoWPairModel(
            vectorizer_params=vectorizer_params,
            model_params=model_params,
            random_state=78_000 + int(outer_fold),
        ).fit(full_pairs)
        importance.update(ridge_model.top_features(top_n))
    except Exception as exc:
        if native_capture_sink is not None:
            raise RuntimeError(
                "native matched-pair proof requires the genuine full Ridge diagnostic fit"
            ) from exc
        logger.debug("Skipping ridge pair-uplift feature table for %s: %s", view_name, exc)
    if native_capture_sink is not None:
        if ridge_model is None:
            raise RuntimeError("native matched-pair proof lacks its full Ridge diagnostic fit")
        native_capture_sink.record_bow_pair_full(
            view_name=view_name,
            view_index=view_index,
            full_pairs=full_pairs,
            offset_model=full_model,
            ridge_model=ridge_model,
        )
    importance.update(
        {
            "view_name": str(view_name),
            "n_matched_training_pairs": int(len(full_pairs)),
            "pair_matching": {
                "propensity_caliper": float(propensity_caliper),
                "outcome_caliper": float(outcome_caliper),
                "max_controls_per_candidate": int(max_controls_per_candidate),
                "nearest_fallback_controls": int(nearest_fallback_controls),
            },
        }
    )
    return {
        "importance": importance,
        "n_matched_training_pairs": int(len(full_pairs)),
    }


def _validate_bow_pair_importance_checkpoint(payload: Dict[str, Any]) -> None:
    if not isinstance(payload.get("importance"), dict):
        raise ValueError("BoW pair importance checkpoint is invalid")
    if int(payload.get("n_matched_training_pairs", -1)) < 0:
        raise ValueError("BoW pair importance checkpoint has an invalid pair count")


def fit_bow_pair_uplift_train_test(
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
    vectorizer_params: Dict[str, Any],
    model_params: Dict[str, Any],
    outer_fold: int,
    view_name: str,
    view_index: int,
    effect_folds: int,
    propensity_caliper: float,
    outcome_caliper: float,
    max_controls_per_candidate: int,
    nearest_fallback_controls: int,
    l2_alpha: float,
    max_iter: int,
    top_n: int,
    native_capture_sink: Optional[Any] = None,
    checkpoint_store: Optional[Stage1CheckpointStore] = None,
    checkpoint_dependencies: Sequence[str] = (),
    checkpoint_namespace: Optional[str] = None,
) -> PairUpliftFitResult:
    y_train = np.asarray(y_train, dtype=float)
    t_train = np.asarray(t_train, dtype=float)
    e_train = np.asarray(e_train, dtype=float)
    m_train = np.asarray(m_train, dtype=float)
    e_test = np.asarray(e_test, dtype=float)
    m_test = np.asarray(m_test, dtype=float)
    folds = _bounded_folds(effect_folds, len(train_df))
    splitter = KFold(
        n_splits=folds,
        shuffle=True,
        random_state=91_000 + 100 * int(outer_fold) + 1_000 * int(view_index),
    )

    train_delta = np.full(len(train_df), np.nan, dtype=float)
    train_prob = np.full(len(train_df), np.nan, dtype=float)
    train_n_controls = np.zeros(len(train_df), dtype=float)
    test_deltas = []
    test_probs = []
    test_n_controls = []
    evidence_rows: List[Dict[str, Any]] = []
    prediction_frames = []

    for inner_fold, (fit_pos, heldout_pos) in enumerate(splitter.split(train_df), start=1):
        fit_pos = np.asarray(fit_pos, dtype=int)
        heldout_pos = np.asarray(heldout_pos, dtype=int)

        def compute_fold() -> Dict[str, Any]:
            return _fit_bow_pair_uplift_fold(
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
                vectorizer_params=vectorizer_params,
                outer_fold=outer_fold,
                view_name=view_name,
                view_index=view_index,
                inner_fold=inner_fold,
                fit_pos=fit_pos,
                heldout_pos=heldout_pos,
                propensity_caliper=propensity_caliper,
                outcome_caliper=outcome_caliper,
                max_controls_per_candidate=max_controls_per_candidate,
                nearest_fallback_controls=nearest_fallback_controls,
                l2_alpha=l2_alpha,
                max_iter=max_iter,
                native_capture_sink=native_capture_sink,
            )

        if (
            checkpoint_store is None
            or checkpoint_namespace is None
            or native_capture_sink is not None
        ):
            fold_result = compute_fold()
        else:
            fold_result = checkpoint_store.run(
                f"{checkpoint_namespace}/fold_{int(inner_fold):03d}",
                compute_fold,
                parameters={
                    "outer_fold": int(outer_fold),
                    "inner_fold": int(inner_fold),
                    "total_folds": int(folds),
                    "view_name": str(view_name),
                    "view_index": int(view_index),
                    "propensity_caliper": float(propensity_caliper),
                    "outcome_caliper": float(outcome_caliper),
                    "max_controls_per_candidate": int(max_controls_per_candidate),
                    "nearest_fallback_controls": int(nearest_fallback_controls),
                    "l2_alpha": float(l2_alpha),
                    "max_iter": int(max_iter),
                },
                dependencies=tuple(checkpoint_dependencies),
                rows={
                    "fit": row_id_identity(train_df.iloc[fit_pos]["_oci_row_id"].to_numpy()),
                    "validation": row_id_identity(
                        train_df.iloc[heldout_pos]["_oci_row_id"].to_numpy()
                    ),
                    "heldout": row_id_identity(test_df["_oci_row_id"].to_numpy()),
                },
                validator=lambda payload: _validate_pair_fold_checkpoint(
                    payload,
                    heldout_rows=len(heldout_pos),
                    test_rows=len(test_df),
                ),
            ).value
        result_heldout_pos = np.asarray(fold_result["heldout_pos"], dtype=int)
        train_delta[result_heldout_pos] = fold_result["fold_delta"]
        train_prob[result_heldout_pos] = fold_result["fold_prob"]
        train_n_controls[result_heldout_pos] = fold_result["fold_n"]
        test_deltas.append(np.asarray(fold_result["test_delta"], dtype=float))
        test_probs.append(np.asarray(fold_result["test_prob"], dtype=float))
        test_n_controls.append(np.asarray(fold_result["test_n"], dtype=float))
        evidence_rows.append(fold_result["evidence"])
        if not fold_result["prediction_frame"].empty:
            prediction_frames.append(fold_result["prediction_frame"])

    test_delta = np.nanmean(np.vstack(test_deltas), axis=0) if test_deltas else np.nan
    test_prob = np.nanmean(np.vstack(test_probs), axis=0) if test_probs else np.nan
    test_n = np.nanmean(np.vstack(test_n_controls), axis=0) if test_n_controls else np.nan

    def compute_full_importance() -> Dict[str, Any]:
        return _fit_bow_pair_full_importance(
            train_df=train_df,
            texts_train=texts_train,
            y_train=y_train,
            t_train=t_train,
            e_train=e_train,
            m_train=m_train,
            vectorizer_params=vectorizer_params,
            model_params=model_params,
            outer_fold=outer_fold,
            view_name=view_name,
            view_index=view_index,
            propensity_caliper=propensity_caliper,
            outcome_caliper=outcome_caliper,
            max_controls_per_candidate=max_controls_per_candidate,
            nearest_fallback_controls=nearest_fallback_controls,
            l2_alpha=l2_alpha,
            max_iter=max_iter,
            top_n=top_n,
            native_capture_sink=native_capture_sink,
        )

    if (
        checkpoint_store is None
        or checkpoint_namespace is None
        or native_capture_sink is not None
    ):
        importance_payload = compute_full_importance()
    else:
        fold_dependencies = tuple(
            f"{checkpoint_namespace}/fold_{inner_fold:03d}"
            for inner_fold in range(1, folds + 1)
        )
        importance_payload = checkpoint_store.run(
            f"{checkpoint_namespace}/full_importance",
            compute_full_importance,
            parameters={
                "outer_fold": int(outer_fold),
                "view_name": str(view_name),
                "view_index": int(view_index),
                "top_n": int(top_n),
            },
            dependencies=fold_dependencies,
            rows={"fit": row_id_identity(train_df["_oci_row_id"].to_numpy())},
            validator=_validate_bow_pair_importance_checkpoint,
        ).value
    importance = dict(importance_payload["importance"])
    n_matched_training_pairs = int(importance_payload["n_matched_training_pairs"])
    treated_eval = (t_train.astype(int) == 1) & np.isfinite(train_prob)
    metrics = {
        "n_train_matched_pairs": n_matched_training_pairs,
        "train_candidate_control_mean": _finite_or_none(np.nanmean(train_n_controls)),
        "test_candidate_control_mean": _finite_or_none(np.nanmean(test_n)),
        "treated_oof": _binary_metrics(y_train[treated_eval], train_prob[treated_eval]),
    }
    if prediction_frames:
        pair_predictions = pd.concat(prediction_frames, ignore_index=True)
    else:
        pair_predictions = pd.DataFrame()
    return PairUpliftFitResult(
        train_delta_logit=train_delta,
        test_delta_logit=np.asarray(test_delta, dtype=float),
        train_pred_prob=train_prob,
        test_pred_prob=np.asarray(test_prob, dtype=float),
        train_n_controls=train_n_controls,
        test_n_controls=np.asarray(test_n, dtype=float),
        feature_importance=importance,
        evidence_rows=evidence_rows,
        attention_rows=[],
        prediction_frame=pair_predictions,
        metrics=metrics,
    )


class HTRPairUpliftNet(nn.Module):
    def __init__(
        self,
        extractor: nn.Module,
        hidden_dim: int,
        dropout: float = 0.1,
        *,
        head_depth: int = 2,
        head_activation: str = "relu",
        head_layer_norm: bool = False,
        head_bias: bool = True,
    ):
        super().__init__()
        self.extractor = extractor
        dim = int(extractor.output_dim)
        if isinstance(head_depth, bool) or int(head_depth) < 1:
            raise ValueError("HTR pair head depth must be positive")
        if int(hidden_dim) < 1 or not 0.0 <= float(dropout) < 1.0:
            raise ValueError("HTR pair head dimension/dropout is invalid")
        if type(head_layer_norm) is not bool or type(head_bias) is not bool:
            raise TypeError("HTR pair head norm/bias must be exact booleans")
        activation_factories = {
            "gelu_exact": lambda: nn.GELU(approximate="none"),
            "gelu_tanh": lambda: nn.GELU(approximate="tanh"),
            "relu": lambda: nn.ReLU(inplace=False),
            "silu": lambda: nn.SiLU(inplace=False),
            "tanh": nn.Tanh,
        }
        if str(head_activation) not in activation_factories:
            raise ValueError("HTR pair head activation is unsupported")
        self._head_configuration = {
            "hidden_dim": int(hidden_dim),
            "depth": int(head_depth),
            "activation": str(head_activation),
            "dropout": float(dropout),
            "layer_norm": head_layer_norm,
            "bias": head_bias,
        }
        layers: list[nn.Module] = []
        source_dim = 4 * dim
        for _ in range(int(head_depth)):
            layers.append(
                nn.Linear(source_dim, int(hidden_dim), bias=head_bias)
            )
            if head_layer_norm:
                layers.append(nn.LayerNorm(int(hidden_dim)))
            layers.append(activation_factories[str(head_activation)]())
            layers.append(nn.Dropout(float(dropout), inplace=False))
            source_dim = int(hidden_dim)
        layers.append(nn.Linear(source_dim, 1, bias=head_bias))
        self.fusion = nn.Sequential(*layers)

    def head_configuration(self) -> Dict[str, Any]:
        return dict(self._head_configuration)

    def forward(self, control_texts: Sequence[str], treated_texts: Sequence[str]) -> torch.Tensor:
        control = self.extractor(list(control_texts))
        treated = self.extractor(list(treated_texts))
        fused = torch.cat([control, treated, treated - control, torch.abs(treated - control)], dim=1)
        return self.fusion(fused).squeeze(-1)


def _iter_batches(n_rows: int, batch_size: int, *, shuffle: bool, seed: int):
    order = np.arange(n_rows, dtype=int)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(order)
    for start in range(0, n_rows, max(1, int(batch_size))):
        yield order[start:start + max(1, int(batch_size))]


def _train_htr_pair_model(
    *,
    runner: Any,
    pairs: pd.DataFrame,
    outer_fold: int,
    inner_fold: int,
    total_folds: int,
) -> Optional[HTRPairUpliftNet]:
    if pairs.empty or len(np.unique(pairs["label"].to_numpy(dtype=int))) < 2:
        return None
    extractor = runner._create_extractor()
    hidden_dim = int(getattr(runner.config.architecture, "htr_prediction_head_hidden_dim", 64))
    model = HTRPairUpliftNet(extractor=extractor, hidden_dim=hidden_dim).to(runner.device)
    model.extractor.fit_tokenizer(
        pairs["control_text"].astype(str).tolist() + pairs["treated_text"].astype(str).tolist()
    )
    train_config = runner.config.training
    epochs = runner._effect_epochs()
    batch_size = getattr(train_config, "effect_batch_size", None)
    if batch_size is None:
        batch_size = train_config.batch_size
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=train_config.learning_rate,
        weight_decay=getattr(train_config, "weight_decay", 0.01),
    )
    logger.info(
        "Outer fold %s HTR pair-uplift fold %s/%s: pairs=%s epochs=%s batch_size=%s",
        outer_fold,
        inner_fold,
        total_folds,
        len(pairs),
        epochs,
        batch_size,
    )
    y = torch.as_tensor(pairs["label"].to_numpy(dtype=np.float32), device=runner.device)
    base = torch.as_tensor(pairs["base_logit"].to_numpy(dtype=np.float32), device=runner.device)
    control_text = pairs["control_text"].astype(str).tolist()
    treated_text = pairs["treated_text"].astype(str).tolist()
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_pos in _iter_batches(
            len(pairs),
            int(batch_size),
            shuffle=True,
            seed=55_000 + 100 * int(epoch) + int(inner_fold),
        ):
            optimizer.zero_grad(set_to_none=True)
            delta = model(
                [control_text[int(pos)] for pos in batch_pos],
                [treated_text[int(pos)] for pos in batch_pos],
            )
            logits = base[batch_pos] + delta
            loss = F.binary_cross_entropy_with_logits(logits, y[batch_pos])
            loss.backward()
            optimizer.step()
    return model


def _predict_htr_pair_delta(
    *,
    runner: Any,
    model: Optional[HTRPairUpliftNet],
    pairs: pd.DataFrame,
) -> np.ndarray:
    if pairs.empty:
        return np.zeros(0, dtype=float)
    if model is None:
        return np.zeros(len(pairs), dtype=float)
    batch_size = getattr(runner.config.training, "effect_batch_size", None)
    if batch_size is None:
        batch_size = runner.config.training.batch_size
    control_text = pairs["control_text"].astype(str).tolist()
    treated_text = pairs["treated_text"].astype(str).tolist()
    outputs = []
    model.eval()
    with torch.no_grad():
        for batch_pos in _iter_batches(len(pairs), int(batch_size), shuffle=False, seed=0):
            outputs.append(
                model(
                    [control_text[int(pos)] for pos in batch_pos],
                    [treated_text[int(pos)] for pos in batch_pos],
                )
                .detach()
                .cpu()
                .numpy()
            )
    return np.concatenate(outputs) if outputs else np.zeros(0, dtype=float)


def _htr_pair_attention_rows(
    *,
    runner: Any,
    model: Optional[HTRPairUpliftNet],
    pairs: pd.DataFrame,
    pair_delta: np.ndarray,
    outer_fold: int,
    inner_fold: int,
    max_pairs: int,
) -> List[Dict[str, Any]]:
    if model is None or pairs.empty or int(max_pairs) <= 0:
        return []
    order = np.argsort(-np.abs(np.asarray(pair_delta, dtype=float)))[: int(max_pairs)]
    selected = pairs.iloc[order].reset_index(drop=True)
    selected_delta = np.asarray(pair_delta, dtype=float)[order]
    rows: List[Dict[str, Any]] = []
    for side, text_col, row_col in [
        ("treated_candidate", "treated_text", "treated_row_id"),
        ("matched_control", "control_text", "control_row_id"),
    ]:
        texts = selected[text_col].astype(str).tolist()
        row_ids = selected[row_col].astype(int).tolist()
        metadata = []
        for offset, pair in selected.iterrows():
            delta = float(selected_delta[int(offset)])
            base_logit = float(pair["base_logit"])
            metadata.append(
                {
                    "outer_fold": int(outer_fold),
                    "inner_fold": int(inner_fold),
                    "pair_side": side,
                    "candidate_row_id": int(pair["candidate_row_id"]),
                    "control_row_id": int(pair["control_row_id"]),
                    "pair_delta_logit": delta,
                    "pair_pred_prob": float(expit(base_logit + delta)),
                    "pair_base_prob": float(pair["base_prob"]),
                    "pair_score_abs_diff_sum": float(pair["score_abs_diff_sum"]),
                }
            )
        records = model.extractor.get_attention_evidence(
            texts,
            row_ids=row_ids,
            fold=inner_fold,
            stage="effect_modifier",
            top_k=runner.avf_config.attention_top_k_chunks,
            metadata=metadata,
        )
        for record in records:
            record.setdefault("model_family", "htr_pair_uplift")
            record.setdefault("target_source", "matched_pair_uplift_delta_logit")
        rows.extend(records)
    return rows


def _fit_htr_pair_uplift_fold(
    *,
    runner: Any,
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
    inner_fold: int,
    folds: int,
    fit_pos: np.ndarray,
    heldout_pos: np.ndarray,
    propensity_caliper: float,
    outcome_caliper: float,
    max_controls_per_candidate: int,
    nearest_fallback_controls: int,
    max_attention_pairs: int,
    native_capture_sink: Optional[Any],
) -> Dict[str, Any]:
    """Fit and score one independently checkpointable HTR pair fold."""

    fit_pos = np.asarray(fit_pos, dtype=int)
    heldout_pos = np.asarray(heldout_pos, dtype=int)
    fit_df = train_df.iloc[fit_pos].reset_index(drop=True)
    heldout_df = train_df.iloc[heldout_pos].reset_index(drop=True)
    fit_pairs = build_training_pairs(
        fit_df,
        texts=[texts_train[int(pos)] for pos in fit_pos],
        treatment=t_train[fit_pos],
        outcome=y_train[fit_pos],
        propensity=e_train[fit_pos],
        outcome_prob=m_train[fit_pos],
        propensity_caliper=propensity_caliper,
        outcome_caliper=outcome_caliper,
    )
    model: Optional[HTRPairUpliftNet] = None
    try:
        model = _train_htr_pair_model(
            runner=runner,
            pairs=fit_pairs,
            outer_fold=outer_fold,
            inner_fold=inner_fold,
            total_folds=folds,
        )
        control_pos = fit_pos[t_train[fit_pos].astype(int) == 0]
        control_df = train_df.iloc[control_pos].reset_index(drop=True)
        heldout_pairs = build_candidate_pairs(
            heldout_df,
            control_df,
            candidate_texts=[texts_train[int(pos)] for pos in heldout_pos],
            control_texts=[texts_train[int(pos)] for pos in control_pos],
            candidate_propensity=e_train[heldout_pos],
            candidate_outcome_prob=m_train[heldout_pos],
            control_propensity=e_train[control_pos],
            control_outcome_prob=m_train[control_pos],
            propensity_caliper=propensity_caliper,
            outcome_caliper=outcome_caliper,
            max_controls_per_candidate=max_controls_per_candidate,
            nearest_fallback_controls=nearest_fallback_controls,
        )
        heldout_pair_delta = _predict_htr_pair_delta(
            runner=runner,
            model=model,
            pairs=heldout_pairs,
        )
        fold_delta, fold_prob, fold_n = aggregate_pair_predictions(
            heldout_pairs,
            heldout_pair_delta,
            len(heldout_df),
        )
        attention_rows = _htr_pair_attention_rows(
            runner=runner,
            model=model,
            pairs=heldout_pairs,
            pair_delta=heldout_pair_delta,
            outer_fold=outer_fold,
            inner_fold=inner_fold,
            max_pairs=max_attention_pairs,
        )

        test_pairs = build_candidate_pairs(
            test_df,
            control_df,
            candidate_texts=texts_test,
            control_texts=[texts_train[int(pos)] for pos in control_pos],
            candidate_propensity=e_test,
            candidate_outcome_prob=m_test,
            control_propensity=e_train[control_pos],
            control_outcome_prob=m_train[control_pos],
            propensity_caliper=propensity_caliper,
            outcome_caliper=outcome_caliper,
            max_controls_per_candidate=max_controls_per_candidate,
            nearest_fallback_controls=nearest_fallback_controls,
        )
        test_pair_delta = _predict_htr_pair_delta(
            runner=runner,
            model=model,
            pairs=test_pairs,
        )
        fold_test_delta, fold_test_prob, fold_test_n = aggregate_pair_predictions(
            test_pairs,
            test_pair_delta,
            len(test_df),
        )
        if native_capture_sink is not None:
            native_capture_sink.record_htr_pair_fold(
                fold=inner_fold,
                fit_pos=fit_pos,
                validation_pos=heldout_pos,
                fit_pairs=fit_pairs,
                validation_pairs=heldout_pairs,
                heldout_pairs=test_pairs,
                model=model,
                validation_pair_delta=heldout_pair_delta,
                validation_delta=fold_delta,
                validation_probability=fold_prob,
                validation_n_controls=fold_n,
                heldout_pair_delta=test_pair_delta,
                heldout_delta=fold_test_delta,
                heldout_probability=fold_test_prob,
                heldout_n_controls=fold_test_n,
            )
        treated_eval = (t_train[heldout_pos].astype(int) == 1) & np.isfinite(fold_prob)
        evidence = {
            "outer_fold": int(outer_fold),
            "inner_fold": int(inner_fold),
            "source_family": "htr_pair_uplift",
            "objective": "matched_pair_uplift_delta_logit",
            "target_name": "treated_observed_outcome",
            "train_rows": int(len(fit_pos)),
            "heldout_rows": int(len(heldout_pos)),
            "outer_test_rows": int(len(test_df)),
            "matched_pair_train_rows": int(len(fit_pairs)),
            "heldout_candidate_pair_rows": int(len(heldout_pairs)),
            "outer_test_candidate_pair_rows": int(len(test_pairs)),
            "heldout_treated_auroc": _safe_roc(
                y_train[heldout_pos][treated_eval],
                fold_prob[treated_eval],
            ),
            "prediction_provenance": "inner_fold_pair_model_heldout_and_outer_test",
        }
        prediction_frame = pd.DataFrame()
        if not heldout_pairs.empty:
            prediction_frame = heldout_pairs[["candidate_row_id", "control_row_id"]].copy()
            prediction_frame["outer_fold"] = int(outer_fold)
            prediction_frame["inner_fold"] = int(inner_fold)
            prediction_frame["split_role"] = "train_inner_oof_pairs"
            prediction_frame["source_name"] = "htr__pair_uplift"
            prediction_frame["pair_delta_logit"] = heldout_pair_delta
            prediction_frame["pair_pred_prob"] = expit(
                heldout_pairs["base_logit"].to_numpy(dtype=float) + heldout_pair_delta
            )
        return {
            "heldout_pos": heldout_pos,
            "fold_delta": fold_delta,
            "fold_prob": fold_prob,
            "fold_n": fold_n,
            "test_delta": fold_test_delta,
            "test_prob": fold_test_prob,
            "test_n": fold_test_n,
            "attention_rows": attention_rows,
            "evidence": evidence,
            "prediction_frame": prediction_frame,
        }
    finally:
        if model is not None and hasattr(runner, "_cleanup_model"):
            runner._cleanup_model(model)


def _validate_pair_fold_checkpoint(
    payload: Dict[str, Any],
    *,
    heldout_rows: int,
    test_rows: int,
) -> None:
    if len(np.asarray(payload["heldout_pos"])) != int(heldout_rows):
        raise ValueError("pair checkpoint held-out positions changed")
    for name in ("fold_delta", "fold_prob", "fold_n"):
        if len(np.asarray(payload[name])) != int(heldout_rows):
            raise ValueError(f"pair checkpoint {name} row count changed")
    for name in ("test_delta", "test_prob", "test_n"):
        if len(np.asarray(payload[name])) != int(test_rows):
            raise ValueError(f"pair checkpoint {name} row count changed")
    if not isinstance(payload.get("prediction_frame"), pd.DataFrame):
        raise ValueError("pair checkpoint prediction frame is invalid")


def fit_htr_pair_uplift_train_test(
    *,
    runner: Any,
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
    effect_folds: int,
    propensity_caliper: float,
    outcome_caliper: float,
    max_controls_per_candidate: int,
    nearest_fallback_controls: int,
    max_attention_pairs: int,
    native_capture_sink: Optional[Any] = None,
    checkpoint_store: Optional[Stage1CheckpointStore] = None,
    checkpoint_dependencies: Sequence[str] = (),
) -> PairUpliftFitResult:
    y_train = np.asarray(y_train, dtype=float)
    t_train = np.asarray(t_train, dtype=float)
    e_train = np.asarray(e_train, dtype=float)
    m_train = np.asarray(m_train, dtype=float)
    e_test = np.asarray(e_test, dtype=float)
    m_test = np.asarray(m_test, dtype=float)
    folds = _bounded_folds(effect_folds, len(train_df))
    splitter = KFold(n_splits=folds, shuffle=True, random_state=92_000 + int(outer_fold))

    train_delta = np.full(len(train_df), np.nan, dtype=float)
    train_prob = np.full(len(train_df), np.nan, dtype=float)
    train_n_controls = np.zeros(len(train_df), dtype=float)
    test_deltas = []
    test_probs = []
    test_n_controls = []
    evidence_rows: List[Dict[str, Any]] = []
    attention_rows: List[Dict[str, Any]] = []
    prediction_frames = []

    for inner_fold, (fit_pos, heldout_pos) in enumerate(splitter.split(train_df), start=1):
        fit_pos = np.asarray(fit_pos, dtype=int)
        heldout_pos = np.asarray(heldout_pos, dtype=int)

        def compute_fold() -> Dict[str, Any]:
            return _fit_htr_pair_uplift_fold(
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
                inner_fold=inner_fold,
                folds=folds,
                fit_pos=fit_pos,
                heldout_pos=heldout_pos,
                propensity_caliper=propensity_caliper,
                outcome_caliper=outcome_caliper,
                max_controls_per_candidate=max_controls_per_candidate,
                nearest_fallback_controls=nearest_fallback_controls,
                max_attention_pairs=max_attention_pairs,
                native_capture_sink=native_capture_sink,
            )
        if checkpoint_store is None or native_capture_sink is not None:
            fold_result = compute_fold()
        else:
            fold_result = checkpoint_store.run(
                f"htr/pair_uplift/fold_{int(inner_fold):03d}",
                compute_fold,
                parameters={
                    "outer_fold": int(outer_fold),
                    "inner_fold": int(inner_fold),
                    "total_folds": int(folds),
                    "propensity_caliper": float(propensity_caliper),
                    "outcome_caliper": float(outcome_caliper),
                    "max_controls_per_candidate": int(max_controls_per_candidate),
                    "nearest_fallback_controls": int(nearest_fallback_controls),
                    "max_attention_pairs": int(max_attention_pairs),
                },
                dependencies=tuple(checkpoint_dependencies),
                rows={
                    "fit": row_id_identity(train_df.iloc[fit_pos]["_oci_row_id"].to_numpy()),
                    "validation": row_id_identity(
                        train_df.iloc[heldout_pos]["_oci_row_id"].to_numpy()
                    ),
                    "heldout": row_id_identity(test_df["_oci_row_id"].to_numpy()),
                },
                validator=lambda payload: _validate_pair_fold_checkpoint(
                    payload,
                    heldout_rows=len(heldout_pos),
                    test_rows=len(test_df),
                ),
                deterministic_seed=True,
            ).value
        result_heldout_pos = np.asarray(fold_result["heldout_pos"], dtype=int)
        train_delta[result_heldout_pos] = fold_result["fold_delta"]
        train_prob[result_heldout_pos] = fold_result["fold_prob"]
        train_n_controls[result_heldout_pos] = fold_result["fold_n"]
        test_deltas.append(np.asarray(fold_result["test_delta"], dtype=float))
        test_probs.append(np.asarray(fold_result["test_prob"], dtype=float))
        test_n_controls.append(np.asarray(fold_result["test_n"], dtype=float))
        attention_rows.extend(fold_result["attention_rows"])
        evidence_rows.append(fold_result["evidence"])
        if not fold_result["prediction_frame"].empty:
            prediction_frames.append(fold_result["prediction_frame"])

    test_delta = np.nanmean(np.vstack(test_deltas), axis=0) if test_deltas else np.nan
    test_prob = np.nanmean(np.vstack(test_probs), axis=0) if test_probs else np.nan
    test_n = np.nanmean(np.vstack(test_n_controls), axis=0) if test_n_controls else np.nan
    treated_eval = (t_train.astype(int) == 1) & np.isfinite(train_prob)
    metrics = {
        "train_candidate_control_mean": _finite_or_none(np.nanmean(train_n_controls)),
        "test_candidate_control_mean": _finite_or_none(np.nanmean(test_n)),
        "treated_oof": _binary_metrics(y_train[treated_eval], train_prob[treated_eval]),
    }
    prediction_frame = (
        pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    )
    return PairUpliftFitResult(
        train_delta_logit=train_delta,
        test_delta_logit=np.asarray(test_delta, dtype=float),
        train_pred_prob=train_prob,
        test_pred_prob=np.asarray(test_prob, dtype=float),
        train_n_controls=train_n_controls,
        test_n_controls=np.asarray(test_n, dtype=float),
        feature_importance={
            "source_family": "htr_pair_uplift",
            "attention_rows": int(len(attention_rows)),
            "pair_matching": {
                "propensity_caliper": float(propensity_caliper),
                "outcome_caliper": float(outcome_caliper),
                "max_controls_per_candidate": int(max_controls_per_candidate),
                "nearest_fallback_controls": int(nearest_fallback_controls),
            },
        },
        evidence_rows=evidence_rows,
        attention_rows=attention_rows,
        prediction_frame=prediction_frame,
        metrics=metrics,
    )
