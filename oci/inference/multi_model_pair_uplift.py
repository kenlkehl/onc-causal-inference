"""Matched-pair effect-modification lane (``matched_pair_uplift``).

Treated and untreated rows of one Stage 1 training context are matched 1:1,
without replacement, within calipers on two cross-fitted scores: the
propensity ``e`` and the prognostic score ``mu0`` (the untreated-arm outcome
model, a probability for binary outcomes and a mean for continuous ones). The
matching is a maximum bipartite matching (Hopcroft-Karp) whose candidate lists
are ordered closest first, so each treated row prefers its closest eligible
control. Rows outside the calipers are never matched.

Each pair is described by the mean of its two members' bag-of-words vectors,
one block of features per pair:

* **Binary outcomes** keep only the discordant pairs (exactly one member had
  the event) and fit a conditional logistic regression of
  ``D = 1{the treated member had the event}`` with the fixed offset
  ``logit mu0(x_t) - logit mu0(x_c)``. The intercept is the average
  conditional log odds ratio; the coefficients are log-odds-ratio-scale
  effect modification.
* **Continuous outcomes** keep every pair and regress the adjusted paired
  difference ``(Y_t - Y_c) - (mu0(x_t) - mu0(x_c))``. The intercept is the
  average effect; the coefficients are additive effect modification.

Both fits are elastic nets on standardized pair features with an unpenalized
intercept. The penalty is chosen by K-fold cross-validation over pairs on the
held-out conditional log-likelihood (binary) or squared error (continuous).
A nested cross-fit over the same pair folds reports the held-out gain of the
fitted model over the intercept-only (constant-effect) model with the same
offset; it is reported, never used as a gate.

Callers pass only the context's training rows for matching and fitting;
test rows are used only to compute ``tau(x)`` from each patient's own
features after the fit.
"""

from __future__ import annotations

import json
import logging
import math
import warnings
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from ..config import BoWViewConfig
from .sparse_text_modeling import bow_vectorizer_params, make_bow_vectorizer
from .stage1_architectures import MATCHED_PAIR_EVIDENCE_SCHEMA_VERSION
from .stage1_checkpointing import lightweight_fingerprint

logger = logging.getLogger(__name__)

# Versions every checkpoint unit of this lane; earlier pair units never match.
MATCHED_PAIR_CHECKPOINT_VERSION = "matched_pair_conditional_logit_checkpoint_v1"

PROPENSITY_SOURCE = "bow_nuisance_out_of_fold_mean"
BINARY_PROGNOSTIC_SOURCE = "dina_crossfit_untreated_outcome_probability"
CONTINUOUS_PROGNOSTIC_SOURCE = "bow_elastic_net_untreated_outcome_crossfit"
DEFAULT_BINARY_PROGNOSTIC_CALIPER = 0.05
DEFAULT_CONTINUOUS_PROGNOSTIC_CALIPER_SD = 0.2
# Each treated row keeps at most this many eligible controls, closest first,
# so the matching graph stays tractable in large cohorts.
MAX_CANDIDATE_CONTROLS = 256
PENALTY_GRID_SIZE = 30
_PROBABILITY_EPS = 1e-6
_KKT_RELATIVE_TOLERANCE = 1e-4
_SD_FLOOR = 1e-12
_DENSE_RESTRICTED_LIMIT = 4_000_000
_TERM_FIELDS = ("effect_modification_positive", "effect_modification_negative")


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchedPairSettings:
    """Scientific settings of the lane, resolved from the Stage 1 template."""

    outcome_type: str
    propensity_caliper: float
    prognostic_caliper: Optional[float]
    l1_ratio: float
    cv_folds: int
    min_discordant_pairs: int
    min_pairs: int
    max_iter: int
    top_n: int

    @classmethod
    def from_config(cls, config: Any, *, outcome_type: str, top_n: int) -> "MatchedPairSettings":
        outcome = str(outcome_type).strip().lower()
        if outcome not in {"binary", "continuous"}:
            raise ValueError("matched_pair_uplift requires a binary or continuous outcome")
        caliper = getattr(config, "matched_pair_prognostic_caliper", None)
        return cls(
            outcome_type=outcome,
            propensity_caliper=float(config.matched_pair_propensity_caliper),
            prognostic_caliper=None if caliper is None else float(caliper),
            l1_ratio=float(config.matched_pair_l1_ratio),
            cv_folds=int(config.matched_pair_cv_folds),
            min_discordant_pairs=int(config.matched_pair_min_discordant_pairs),
            min_pairs=int(config.matched_pair_min_pairs),
            max_iter=int(config.matched_pair_max_iter),
            top_n=int(top_n),
        )

    @property
    def binary(self) -> bool:
        return self.outcome_type == "binary"

    def identity(self) -> Dict[str, Any]:
        return {"checkpoint_version": MATCHED_PAIR_CHECKPOINT_VERSION, **asdict(self)}


def resolve_prognostic_caliper(
    settings: MatchedPairSettings,
    prognostic: np.ndarray,
) -> Dict[str, Any]:
    """Return the absolute mu0 caliper and how it was derived.

    Binary: absolute on the probability scale (default 0.05). Continuous: in
    standard deviations of mu0 over the training rows (default 0.2).
    """

    if settings.binary:
        value = (
            DEFAULT_BINARY_PROGNOSTIC_CALIPER
            if settings.prognostic_caliper is None
            else float(settings.prognostic_caliper)
        )
        if not 0.0 < value <= 1.0:
            raise ValueError(
                "matched_pair_prognostic_caliper must be in (0, 1] for a binary outcome"
            )
        return {
            "configured": settings.prognostic_caliper,
            "value": value,
            "scale": "probability",
            "absolute": value,
            "prognostic_score_sd": None,
        }
    value = (
        DEFAULT_CONTINUOUS_PROGNOSTIC_CALIPER_SD
        if settings.prognostic_caliper is None
        else float(settings.prognostic_caliper)
    )
    finite = np.asarray(prognostic, dtype=float)
    finite = finite[np.isfinite(finite)]
    sd = float(np.std(finite)) if finite.size > 1 else 0.0
    return {
        "configured": settings.prognostic_caliper,
        "value": value,
        "scale": "prognostic_score_standard_deviations",
        "absolute": value * sd,
        "prognostic_score_sd": sd,
    }


def matched_pair_seed(base_seed: Any, outer_fold: int) -> int:
    """Deterministic lane seed from the run seed and the context."""

    sequence = np.random.SeedSequence(
        [int(base_seed or 0) & 0xFFFFFFFF, 41_017, int(outer_fold) & 0xFFFFFFFF]
    )
    return int(sequence.generate_state(1)[0] % (2**31 - 1))


def distinct_vectorizer_groups(views: Sequence[BoWViewConfig]) -> List[Dict[str, Any]]:
    """Group the configured BoW views by identical vectorizer settings.

    The pair model is always the elastic net below, so views that differ only
    in their learner would refit the same model; one fit per vectorizer.
    """

    groups: Dict[str, Dict[str, Any]] = {}
    for view_index, view in enumerate(views):
        params = bow_vectorizer_params(view)
        key = json.dumps(params, sort_keys=True, default=str)
        group = groups.get(key)
        if group is None:
            groups[key] = {
                "group_index": len(groups),
                "name": str(view.name),
                "view_index": int(view_index),
                "view_names": [str(view.name)],
                "vectorizer_params": params,
            }
        else:
            group["view_names"].append(str(view.name))
    return list(groups.values())


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def hopcroft_karp(adjacency: Sequence[Sequence[int]], n_right: int) -> Tuple[List[int], List[int]]:
    """Maximum bipartite matching; earlier adjacency entries are tried first."""

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

    def augment(root: int) -> bool:
        # Iterative layered depth-first search; frames are
        # [left, next adjacency position, right taken by this frame].
        stack: List[List[int]] = [[root, 0, -1]]
        while stack:
            frame = stack[-1]
            left = frame[0]
            edges = adjacency[left]
            descended = False
            while frame[1] < len(edges):
                right = edges[frame[1]]
                frame[1] += 1
                matched_left = pair_right[right]
                if matched_left == -1:
                    frame[2] = right
                    for frame_left, _position, frame_right in stack:
                        pair_left[frame_left] = frame_right
                        pair_right[frame_right] = frame_left
                    return True
                if dist[matched_left] == dist[left] + 1:
                    frame[2] = right
                    stack.append([matched_left, 0, -1])
                    descended = True
                    break
            if descended:
                continue
            dist[left] = -1
            stack.pop()
        return False

    while bfs():
        for left in range(n_left):
            if pair_left[left] == -1:
                augment(left)
    return pair_left, pair_right


_PAIR_COLUMNS = [
    "pair_index",
    "treated_pos",
    "control_pos",
    "treated_row_id",
    "control_row_id",
    "propensity_treated",
    "propensity_control",
    "prognostic_treated",
    "prognostic_control",
    "abs_delta_propensity",
    "abs_delta_prognostic",
]


def build_matched_pairs(
    *,
    row_ids: Sequence[Any],
    treatment: np.ndarray,
    propensity: np.ndarray,
    prognostic: np.ndarray,
    propensity_caliper: float,
    prognostic_caliper: float,
    max_candidates: int = MAX_CANDIDATE_CONTROLS,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Match treated to untreated rows 1:1 within both calipers.

    All inputs describe the same rows (one Stage 1 training context).
    Positions in the returned frame index those rows. There is no fallback:
    a treated row without an eligible control stays unmatched.
    """

    row_ids = np.asarray(row_ids)
    t = np.asarray(treatment, dtype=float)
    e = np.asarray(propensity, dtype=float)
    m = np.asarray(prognostic, dtype=float)
    if not (len(row_ids) == len(t) == len(e) == len(m)):
        raise ValueError("matching inputs must describe the same rows")
    propensity_caliper = float(propensity_caliper)
    prognostic_caliper = float(prognostic_caliper)
    if propensity_caliper <= 0.0 or prognostic_caliper < 0.0:
        raise ValueError("matching calipers must be positive")
    eligible = np.isfinite(e) & np.isfinite(m) & np.isfinite(t)
    treated = np.flatnonzero(eligible & (t == 1))
    control = np.flatnonzero(eligible & (t == 0))
    stats: Dict[str, Any] = {
        "n_rows": int(len(t)),
        "n_treated": int(np.sum(t == 1)),
        "n_control": int(np.sum(t == 0)),
        "n_rows_without_scores": int(np.sum(~eligible)),
        "propensity_caliper": propensity_caliper,
        "prognostic_caliper_absolute": prognostic_caliper,
        "max_candidate_controls": int(max_candidates),
        "n_candidate_lists_truncated": 0,
        "nearest_neighbour_fallback": False,
    }
    if not len(treated) or not len(control):
        stats.update({"n_pairs": 0, "n_unmatched_treated": int(len(treated))})
        return pd.DataFrame(columns=_PAIR_COLUMNS), stats

    order = np.argsort(e[control], kind="mergesort")
    sorted_control = control[order]
    control_e = e[sorted_control]
    control_m = m[sorted_control]
    control_ids = row_ids[sorted_control]
    slack = 1e-12
    adjacency: List[List[int]] = []
    truncated = 0
    for position in treated:
        low = int(np.searchsorted(control_e, e[position] - propensity_caliper - slack, "left"))
        high = int(np.searchsorted(control_e, e[position] + propensity_caliper + slack, "right"))
        if high <= low:
            adjacency.append([])
            continue
        delta_e = np.abs(control_e[low:high] - e[position])
        delta_m = np.abs(control_m[low:high] - m[position])
        within = np.flatnonzero(
            (delta_e <= propensity_caliper) & (delta_m <= prognostic_caliper)
        )
        if not within.size:
            adjacency.append([])
            continue
        distance = delta_e[within] / propensity_caliper
        if prognostic_caliper > 0.0:
            distance = distance + delta_m[within] / prognostic_caliper
        ranked = np.lexsort(
            (control_ids[low + within], delta_e[within], distance)
        )
        if len(ranked) > int(max_candidates):
            truncated += 1
            ranked = ranked[: int(max_candidates)]
        adjacency.append((low + within[ranked]).astype(int).tolist())
    pair_left, _pair_right = hopcroft_karp(adjacency, len(sorted_control))

    records = []
    for left, right in enumerate(pair_left):
        if right < 0:
            continue
        tp = int(treated[left])
        cp = int(sorted_control[right])
        records.append(
            {
                "pair_index": len(records),
                "treated_pos": tp,
                "control_pos": cp,
                "treated_row_id": _plain_scalar(row_ids[tp]),
                "control_row_id": _plain_scalar(row_ids[cp]),
                "propensity_treated": float(e[tp]),
                "propensity_control": float(e[cp]),
                "prognostic_treated": float(m[tp]),
                "prognostic_control": float(m[cp]),
                "abs_delta_propensity": float(abs(e[tp] - e[cp])),
                "abs_delta_prognostic": float(abs(m[tp] - m[cp])),
            }
        )
    pairs = pd.DataFrame(records, columns=_PAIR_COLUMNS)
    stats.update(
        {
            "n_pairs": int(len(pairs)),
            "n_unmatched_treated": int(len(treated) - len(pairs)),
            "n_candidate_lists_truncated": int(truncated),
        }
    )
    return pairs, stats


def _plain_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _finite_or_none(value: Any) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def summarize_pairs(
    pairs: pd.DataFrame,
    outcome: np.ndarray,
    *,
    binary: bool,
) -> Dict[str, Any]:
    """Pair counts, balance, and the unadjusted paired effect estimate."""

    outcome = np.asarray(outcome, dtype=float)
    n_pairs = int(len(pairs))
    summary: Dict[str, Any] = {"n_pairs": n_pairs}
    if n_pairs:
        delta_e = pairs["abs_delta_propensity"].to_numpy(dtype=float)
        delta_m = pairs["abs_delta_prognostic"].to_numpy(dtype=float)
        summary.update(
            {
                "mean_abs_delta_propensity": float(np.mean(delta_e)),
                "max_abs_delta_propensity": float(np.max(delta_e)),
                "mean_abs_delta_prognostic": float(np.mean(delta_m)),
                "max_abs_delta_prognostic": float(np.max(delta_m)),
            }
        )
    else:
        summary.update(
            {
                "mean_abs_delta_propensity": None,
                "max_abs_delta_propensity": None,
                "mean_abs_delta_prognostic": None,
                "max_abs_delta_prognostic": None,
            }
        )
    y_t = outcome[pairs["treated_pos"].to_numpy(dtype=int)] if n_pairs else np.zeros(0)
    y_c = outcome[pairs["control_pos"].to_numpy(dtype=int)] if n_pairs else np.zeros(0)
    if binary:
        n10 = int(np.sum((y_t == 1) & (y_c == 0)))
        n01 = int(np.sum((y_t == 0) & (y_c == 1)))
        summary.update(
            {
                "n_discordant_pairs": n10 + n01,
                "n10_treated_event_only": n10,
                "n01_control_event_only": n01,
                "n11_both_events": int(np.sum((y_t == 1) & (y_c == 1))),
                "n00_no_events": int(np.sum((y_t == 0) & (y_c == 0))),
                # Conditional maximum-likelihood odds ratio for 1:1 pairs.
                "mcnemar_log_odds_ratio": (
                    float(np.log(n10 / n01)) if n10 > 0 and n01 > 0 else None
                ),
                "mcnemar_log_odds_ratio_se": (
                    float(np.sqrt(1.0 / n10 + 1.0 / n01)) if n10 > 0 and n01 > 0 else None
                ),
            }
        )
    else:
        raw = y_t - y_c
        adjusted = raw - (
            pairs["prognostic_treated"].to_numpy(dtype=float)
            - pairs["prognostic_control"].to_numpy(dtype=float)
        ) if n_pairs else np.zeros(0)
        summary.update(
            {
                "mean_raw_paired_difference": _finite_or_none(np.mean(raw)) if n_pairs else None,
                "mean_adjusted_paired_difference": (
                    _finite_or_none(np.mean(adjusted)) if n_pairs else None
                ),
                "sd_adjusted_paired_difference": (
                    _finite_or_none(np.std(adjusted, ddof=1)) if n_pairs > 1 else None
                ),
                "se_mean_adjusted_paired_difference": (
                    _finite_or_none(np.std(adjusted, ddof=1) / np.sqrt(n_pairs))
                    if n_pairs > 1
                    else None
                ),
            }
        )
    return summary


@dataclass
class PairAnalysisData:
    """The pairs that inform the fit, with their target and fixed offset."""

    pairs: pd.DataFrame
    target: np.ndarray
    offset: np.ndarray
    loss: str


def pair_analysis_data(
    pairs: pd.DataFrame,
    outcome: np.ndarray,
    *,
    binary: bool,
) -> PairAnalysisData:
    """Binary: discordant pairs, D and the logit-mu0 offset. Continuous: all pairs."""

    outcome = np.asarray(outcome, dtype=float)
    if pairs.empty:
        empty = np.zeros(0, dtype=float)
        return PairAnalysisData(
            pairs.reset_index(drop=True),
            empty,
            empty.copy(),
            "logistic" if binary else "squared_error",
        )
    y_t = outcome[pairs["treated_pos"].to_numpy(dtype=int)]
    y_c = outcome[pairs["control_pos"].to_numpy(dtype=int)]
    mu_t = pairs["prognostic_treated"].to_numpy(dtype=float)
    mu_c = pairs["prognostic_control"].to_numpy(dtype=float)
    if binary:
        discordant = y_t != y_c
        subset = pairs.loc[discordant].reset_index(drop=True)
        target = y_t[discordant].astype(float)
        offset = probability_logit(mu_t[discordant]) - probability_logit(mu_c[discordant])
        return PairAnalysisData(subset, target, offset, "logistic")
    target = (y_t - y_c) - (mu_t - mu_c)
    return PairAnalysisData(
        pairs.reset_index(drop=True),
        target.astype(float),
        np.zeros(len(pairs), dtype=float),
        "squared_error",
    )


def probability_logit(prob: Any) -> np.ndarray:
    p = np.clip(np.asarray(prob, dtype=float), _PROBABILITY_EPS, 1.0 - _PROBABILITY_EPS)
    return logit(p)


# ---------------------------------------------------------------------------
# Elastic net with an offset and an unpenalized intercept
# ---------------------------------------------------------------------------


def _pointwise_loss(eta: np.ndarray, y: np.ndarray, loss: str) -> np.ndarray:
    if loss == "logistic":
        return np.logaddexp(0.0, eta) - y * eta
    return 0.5 * np.square(y - eta)


def _residual(eta: np.ndarray, y: np.ndarray, loss: str) -> np.ndarray:
    if loss == "logistic":
        return expit(eta) - y
    return eta - y


def constant_intercept(y: np.ndarray, offset: np.ndarray, loss: str) -> float:
    """Intercept-only (constant-effect) fit with the fixed offset."""

    y = np.asarray(y, dtype=float)
    offset = np.asarray(offset, dtype=float)
    if not len(y):
        return 0.0
    if loss == "squared_error":
        return float(np.mean(y - offset))
    result = minimize_scalar(
        lambda value: float(np.mean(np.logaddexp(0.0, offset + value) - y * (offset + value))),
        bounds=(-30.0, 30.0),
        method="bounded",
        options={"xatol": 1e-10},
    )
    return float(result.x)


def _full_gradient(
    z: sparse.csc_matrix,
    means: np.ndarray,
    y: np.ndarray,
    offset: np.ndarray,
    intercept: float,
    beta: np.ndarray,
    loss: str,
) -> np.ndarray:
    eta = offset + intercept + np.asarray(z @ beta).ravel() - float(means @ beta)
    residual = _residual(eta, y, loss)
    return np.asarray(z.T @ residual).ravel() / len(y) - means * float(np.mean(residual))


def _soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def _solve_restricted(
    design: Any,
    means: np.ndarray,
    y: np.ndarray,
    offset: np.ndarray,
    intercept: float,
    beta: np.ndarray,
    *,
    l1: float,
    l2: float,
    loss: str,
    max_iter: int,
    tolerance: float,
    step: float,
) -> Tuple[float, np.ndarray, Dict[str, Any]]:
    """FISTA with backtracking and adaptive restart on the active columns.

    Minimizes mean loss + l1 * |beta|_1 + l2 / 2 * |beta|^2 with the linear
    predictor ``offset + intercept + (design - means) @ beta``; stops when the
    KKT residual is at most ``tolerance``.
    """

    n, k = design.shape
    if k == 0:
        return constant_intercept(y, offset, loss), beta, {
            "iterations": 0,
            "converged": True,
            "step": step,
        }
    matrix = design.toarray() if n * k <= _DENSE_RESTRICTED_LIMIT else design.tocsr()

    def smooth(c0: float, c: np.ndarray) -> Tuple[float, float, np.ndarray]:
        eta = offset + c0 + np.asarray(matrix @ c).ravel() - float(means @ c)
        residual = _residual(eta, y, loss)
        g0 = float(np.mean(residual))
        grad = np.asarray(matrix.T @ residual).ravel() / n - means * g0 + l2 * c
        value = float(np.mean(_pointwise_loss(eta, y, loss)) + 0.5 * l2 * float(c @ c))
        return value, g0, grad

    def kkt(c: np.ndarray, g0: float, grad: np.ndarray) -> float:
        residual = np.where(
            c != 0.0,
            np.abs(grad + l1 * np.sign(c)),
            np.maximum(np.abs(grad) - l1, 0.0),
        )
        return max(abs(g0), float(np.max(residual, initial=0.0)))

    x0, x = float(intercept), np.asarray(beta, dtype=float).copy()
    f_x, g0_x, grad_x = smooth(x0, x)
    if kkt(x, g0_x, grad_x) <= tolerance:
        return x0, x, {"iterations": 0, "converged": True, "step": step}
    y0, yv = x0, x.copy()
    f_y, g0_y, grad_y = f_x, g0_x, grad_x
    momentum = 1.0
    for iteration in range(1, int(max_iter) + 1):
        for _ in range(80):
            c0 = y0 - step * g0_y
            c = _soft_threshold(yv - step * grad_y, step * l1)
            f_c, g0_c, grad_c = smooth(c0, c)
            d0 = c0 - y0
            d = c - yv
            bound = f_y + g0_y * d0 + float(grad_y @ d) + (d0 * d0 + float(d @ d)) / (2.0 * step)
            if f_c <= bound + 1e-12 * max(1.0, abs(f_y)):
                break
            step *= 0.5
        else:
            return x0, x, {"iterations": iteration, "converged": False, "step": step}
        if kkt(c, g0_c, grad_c) <= tolerance:
            return c0, c, {"iterations": iteration, "converged": True, "step": step}
        restart = (y0 - c0) * (c0 - x0) + float((yv - c) @ (c - x)) > 0.0
        if restart:
            momentum = 1.0
            y0, yv = c0, c.copy()
            f_y, g0_y, grad_y = f_c, g0_c, grad_c
        else:
            next_momentum = (1.0 + math.sqrt(1.0 + 4.0 * momentum * momentum)) / 2.0
            weight = (momentum - 1.0) / next_momentum
            y0 = c0 + weight * (c0 - x0)
            yv = c + weight * (c - x)
            momentum = next_momentum
            f_y, g0_y, grad_y = smooth(y0, yv)
        x0, x = c0, c
        step *= 1.1
    return x0, x, {"iterations": int(max_iter), "converged": False, "step": step}


def penalty_grid(
    z: sparse.spmatrix,
    means: np.ndarray,
    y: np.ndarray,
    offset: np.ndarray,
    *,
    loss: str,
    l1_ratio: float,
    size: int = PENALTY_GRID_SIZE,
) -> Tuple[np.ndarray, float]:
    """Log-spaced penalties from the smallest one that zeroes every term."""

    n, p = z.shape
    intercept = constant_intercept(y, offset, loss)
    gradient = _full_gradient(sparse.csc_matrix(z), means, y, offset, intercept, np.zeros(p), loss)
    lambda_max = float(np.max(np.abs(gradient), initial=0.0)) / float(l1_ratio)
    if not math.isfinite(lambda_max) or lambda_max <= 1e-12:
        return np.zeros(0, dtype=float), 0.0
    ratio = 1e-2 if n < p else 1e-4
    return lambda_max * np.logspace(0.0, math.log10(ratio), int(size)), lambda_max


def elastic_net_path(
    z: sparse.spmatrix,
    means: np.ndarray,
    y: np.ndarray,
    offset: np.ndarray,
    lambdas: np.ndarray,
    *,
    loss: str,
    l1_ratio: float,
    max_iter: int,
    tolerance: float,
) -> Dict[str, Any]:
    """Warm-started elastic-net path with sequential strong-rule screening.

    The penalty is ``lambda * (l1_ratio * |beta|_1 + (1 - l1_ratio) / 2 *
    |beta|^2)``; the intercept is unpenalized. After each restricted solve,
    the KKT conditions are checked on every column and violators are added.
    """

    z = sparse.csc_matrix(z, dtype=float)
    y = np.asarray(y, dtype=float)
    offset = np.asarray(offset, dtype=float)
    means = np.asarray(means, dtype=float)
    n, p = z.shape
    lambdas = np.asarray(lambdas, dtype=float)
    intercepts = np.zeros(len(lambdas), dtype=float)
    coefficients = np.zeros((len(lambdas), p), dtype=float)
    intercept = constant_intercept(y, offset, loss)
    beta = np.zeros(p, dtype=float)
    diagnostics = {"nonconverged_fits": 0, "maximum_iterations_observed": 0, "kkt_additions": 0}
    if loss == "logistic" and len(np.unique(y)) < 2:
        intercepts[:] = intercept
        diagnostics["constant_reason"] = "single_class"
        return {"intercepts": intercepts, "coefficients": coefficients, "diagnostics": diagnostics}
    gradient = _full_gradient(z, means, y, offset, intercept, beta, loss)
    previous = None
    step = 1.0
    for index, value in enumerate(lambdas):
        l1 = float(value) * float(l1_ratio)
        l2 = float(value) * (1.0 - float(l1_ratio))
        threshold = l1 if previous is None else float(l1_ratio) * (2.0 * value - previous)
        active = np.flatnonzero((np.abs(gradient) >= threshold) | (beta != 0.0))
        while True:
            intercept, active_beta, info = _solve_restricted(
                z[:, active],
                means[active],
                y,
                offset,
                intercept,
                beta[active],
                l1=l1,
                l2=l2,
                loss=loss,
                max_iter=max_iter,
                tolerance=tolerance,
                step=step,
            )
            step = float(info["step"])
            diagnostics["maximum_iterations_observed"] = max(
                diagnostics["maximum_iterations_observed"], int(info["iterations"])
            )
            beta = np.zeros(p, dtype=float)
            beta[active] = active_beta
            gradient = _full_gradient(z, means, y, offset, intercept, beta, loss)
            outside = np.ones(p, dtype=bool)
            outside[active] = False
            violations = np.flatnonzero(outside & (np.abs(gradient) > l1 + tolerance))
            if not violations.size or not info["converged"]:
                if not info["converged"]:
                    diagnostics["nonconverged_fits"] += 1
                break
            diagnostics["kkt_additions"] += int(violations.size)
            active = np.union1d(active, violations)
        intercepts[index] = intercept
        coefficients[index] = beta
        previous = float(value)
    return {"intercepts": intercepts, "coefficients": coefficients, "diagnostics": diagnostics}


def _fold_splits(y: np.ndarray, *, loss: str, folds: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    n = len(y)
    k = max(2, min(int(folds), n // 2))
    placeholder = np.zeros(n)
    if loss == "logistic":
        counts = np.bincount(np.asarray(y, dtype=int), minlength=2)
        if int(counts.min()) >= k:
            splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
            return [(np.asarray(a), np.asarray(b)) for a, b in splitter.split(placeholder, y)]
    splitter = KFold(n_splits=k, shuffle=True, random_state=seed)
    return [(np.asarray(a), np.asarray(b)) for a, b in splitter.split(placeholder)]


def _path_heldout_eta(
    z: sparse.csr_matrix,
    means: np.ndarray,
    y: np.ndarray,
    offset: np.ndarray,
    lambdas: np.ndarray,
    fit: np.ndarray,
    heldout: np.ndarray,
    *,
    loss: str,
    l1_ratio: float,
    max_iter: int,
    tolerance: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    path = elastic_net_path(
        z[fit],
        means,
        y[fit],
        offset[fit],
        lambdas,
        loss=loss,
        l1_ratio=l1_ratio,
        max_iter=max_iter,
        tolerance=tolerance,
    )
    coefficients = path["coefficients"]
    linear = np.asarray(z[heldout] @ coefficients.T) - (means @ coefficients.T)[None, :]
    eta = offset[heldout][:, None] + path["intercepts"][None, :] + linear
    return eta, path["diagnostics"]


def _cross_validated_losses(
    z: sparse.csr_matrix,
    means: np.ndarray,
    y: np.ndarray,
    offset: np.ndarray,
    lambdas: np.ndarray,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    loss: str,
    l1_ratio: float,
    max_iter: int,
    tolerance: float,
) -> Tuple[np.ndarray, List[np.ndarray], Dict[str, int]]:
    """Held-out per-pair loss summed over folds for every penalty."""

    totals = np.zeros(len(lambdas), dtype=float)
    heldout_eta: List[np.ndarray] = []
    diagnostics = {"nonconverged_fits": 0, "maximum_iterations_observed": 0}
    for fit, heldout in splits:
        eta, path_diagnostics = _path_heldout_eta(
            z,
            means,
            y,
            offset,
            lambdas,
            fit,
            heldout,
            loss=loss,
            l1_ratio=l1_ratio,
            max_iter=max_iter,
            tolerance=tolerance,
        )
        heldout_eta.append(eta)
        totals += _pointwise_loss(eta, y[heldout][:, None], loss).sum(axis=0)
        diagnostics["nonconverged_fits"] += int(path_diagnostics["nonconverged_fits"])
        diagnostics["maximum_iterations_observed"] = max(
            diagnostics["maximum_iterations_observed"],
            int(path_diagnostics["maximum_iterations_observed"]),
        )
    return totals, heldout_eta, diagnostics


def fit_penalized_pair_model(
    z: sparse.spmatrix,
    means: np.ndarray,
    y: np.ndarray,
    offset: np.ndarray,
    *,
    loss: str,
    l1_ratio: float,
    cv_folds: int,
    max_iter: int,
    seed: int,
) -> Dict[str, Any]:
    """Choose the penalty by CV over pairs, refit, and cross-fit the gain.

    ``z`` holds standardized pair features and ``means`` their column means;
    the fitted effect is ``intercept + (z - means) @ beta``, so the intercept
    is the effect at the mean pair features.
    """

    z = sparse.csr_matrix(z, dtype=float)
    y = np.asarray(y, dtype=float)
    offset = np.asarray(offset, dtype=float)
    n, p = z.shape
    lambdas, lambda_max = penalty_grid(z, means, y, offset, loss=loss, l1_ratio=l1_ratio)
    tolerance = _KKT_RELATIVE_TOLERANCE * max(lambda_max * l1_ratio, 1e-12)
    splits = _fold_splits(y, loss=loss, folds=cv_folds, seed=seed)
    constant = constant_intercept(y, offset, loss)
    result: Dict[str, Any] = {
        "lambda_max": lambda_max,
        "penalty_grid_size": int(len(lambdas)),
        "lambda_min": float(lambdas[-1]) if len(lambdas) else None,
        "cv_folds": int(len(splits)),
        "constant_intercept": constant,
    }
    if not len(lambdas):
        # No column has any gradient at the constant fit: intercept only.
        result.update(
            {
                "selected_index": None,
                "selected_lambda": None,
                "intercept": constant,
                "beta": np.zeros(p, dtype=float),
                "cv_mean_heldout_loss": [],
                "solver": {"nonconverged_fits": 0, "maximum_iterations_observed": 0},
                "validation": _validation_without_terms(y, offset, splits, loss=loss),
            }
        )
        return result
    totals, outer_eta, cv_diagnostics = _cross_validated_losses(
        z,
        means,
        y,
        offset,
        lambdas,
        splits,
        loss=loss,
        l1_ratio=l1_ratio,
        max_iter=max_iter,
        tolerance=tolerance,
    )
    selected = int(np.argmin(totals))
    final = elastic_net_path(
        z,
        means,
        y,
        offset,
        lambdas[: selected + 1],
        loss=loss,
        l1_ratio=l1_ratio,
        max_iter=max_iter,
        tolerance=tolerance,
    )
    solver = {
        "nonconverged_fits": int(cv_diagnostics["nonconverged_fits"])
        + int(final["diagnostics"]["nonconverged_fits"]),
        "maximum_iterations_observed": max(
            int(cv_diagnostics["maximum_iterations_observed"]),
            int(final["diagnostics"]["maximum_iterations_observed"]),
        ),
    }

    # Nested cross-fit: each outer pair fold's penalty is chosen by an inner
    # CV on that fold's training pairs only; the fold's held-out predictions
    # come from the outer path fitted without it.
    model_loss = np.zeros(n, dtype=float)
    constant_loss = np.zeros(n, dtype=float)
    folds_selecting_largest_penalty = 0
    for fold_index, ((fit, heldout), eta) in enumerate(zip(splits, outer_eta), start=1):
        inner_splits = _fold_splits(
            y[fit], loss=loss, folds=cv_folds, seed=seed + 7_919 * fold_index
        )
        inner_totals, _inner_eta, inner_diagnostics = _cross_validated_losses(
            z[fit],
            means,
            y[fit],
            offset[fit],
            lambdas,
            inner_splits,
            loss=loss,
            l1_ratio=l1_ratio,
            max_iter=max_iter,
            tolerance=tolerance,
        )
        solver["nonconverged_fits"] += int(inner_diagnostics["nonconverged_fits"])
        inner_selected = int(np.argmin(inner_totals))
        folds_selecting_largest_penalty += int(inner_selected == 0)
        model_loss[heldout] = _pointwise_loss(eta[:, inner_selected], y[heldout], loss)
        fold_constant = constant_intercept(y[fit], offset[fit], loss)
        constant_loss[heldout] = _pointwise_loss(
            offset[heldout] + fold_constant, y[heldout], loss
        )
    validation = _validation_summary(
        y,
        model_loss,
        constant_loss,
        loss=loss,
        folds=len(splits),
    )
    validation["folds_selecting_largest_penalty"] = int(folds_selecting_largest_penalty)
    result.update(
        {
            "selected_index": selected,
            "selected_lambda": float(lambdas[selected]),
            "intercept": float(final["intercepts"][-1]),
            "beta": np.asarray(final["coefficients"][-1], dtype=float),
            "cv_mean_heldout_loss": [float(value) / n for value in totals],
            "solver": solver,
            "validation": validation,
        }
    )
    return result


def _validation_summary(
    y: np.ndarray,
    model_loss: np.ndarray,
    constant_loss: np.ndarray,
    *,
    loss: str,
    folds: int,
) -> Dict[str, Any]:
    n = len(y)
    summary: Dict[str, Any] = {
        "method": "nested_pair_cross_fit",
        "folds": int(folds),
        "n_pairs": int(n),
        "baseline": "intercept_only_constant_effect_with_offset",
    }
    if loss == "logistic":
        summary.update(
            {
                "heldout_mean_log_likelihood_model": float(-np.mean(model_loss)),
                "heldout_mean_log_likelihood_constant": float(-np.mean(constant_loss)),
                "heldout_log_likelihood_gain_per_pair": float(
                    np.mean(constant_loss - model_loss)
                ),
                "heldout_log_likelihood_gain_total": float(np.sum(constant_loss - model_loss)),
            }
        )
    else:
        model_mse = float(np.mean(2.0 * model_loss))
        constant_mse = float(np.mean(2.0 * constant_loss))
        summary.update(
            {
                "heldout_mse_model": model_mse,
                "heldout_mse_constant": constant_mse,
                "heldout_mse_reduction": constant_mse - model_mse,
                "heldout_relative_mse_reduction": (
                    (constant_mse - model_mse) / constant_mse if constant_mse > 0 else None
                ),
            }
        )
    return summary


def _validation_without_terms(
    y: np.ndarray,
    offset: np.ndarray,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    loss: str,
) -> Dict[str, Any]:
    constant_loss = np.zeros(len(y), dtype=float)
    for fit, heldout in splits:
        fold_constant = constant_intercept(y[fit], offset[fit], loss)
        constant_loss[heldout] = _pointwise_loss(offset[heldout] + fold_constant, y[heldout], loss)
    summary = _validation_summary(
        y,
        constant_loss,
        constant_loss,
        loss=loss,
        folds=len(splits),
    )
    summary["folds_selecting_largest_penalty"] = int(len(splits))
    return summary


# ---------------------------------------------------------------------------
# One vectorizer configuration
# ---------------------------------------------------------------------------


def _term_rows(
    terms: np.ndarray,
    standardized: np.ndarray,
    raw: np.ndarray,
    top_n: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    nonzero = np.flatnonzero(standardized != 0.0)
    if not nonzero.size:
        return [], []
    order = nonzero[np.lexsort((terms[nonzero], -np.abs(standardized[nonzero])))]
    positive: List[Dict[str, Any]] = []
    negative: List[Dict[str, Any]] = []
    for rank, index in enumerate(order[: int(top_n)], start=1):
        row = {
            "feature": str(terms[index]),
            "score": float(standardized[index]),
            "standardized_coefficient": float(standardized[index]),
            "abs_standardized_coefficient": float(abs(standardized[index])),
            "coefficient": float(raw[index]),
            "rank": int(rank),
        }
        (positive if standardized[index] > 0 else negative).append(row)
    return positive, negative


def fit_matched_pair_view(
    *,
    analysis: PairAnalysisData,
    texts_train: Sequence[str],
    texts_test: Sequence[str],
    group: Mapping[str, Any],
    settings: MatchedPairSettings,
    seed: int,
) -> Dict[str, Any]:
    """Fit the pair model for one vectorizer and predict the test rows.

    The vocabulary and the column scaling are fitted on the analysis pairs'
    member texts (no outcome information); test rows are only transformed.
    """

    view: Dict[str, Any] = {
        "view_name": f"matched_pair__{group['name']}",
        "view_names": list(group["view_names"]),
        "view_index": int(group["view_index"]),
        "vectorizer_config": dict(group["vectorizer_params"]),
        "effect_scale": (
            "conditional_log_odds_ratio" if settings.binary else "outcome_difference"
        ),
        "score_polarity": {"effect_modification": "signed"},
        "n_analysis_pairs": int(len(analysis.pairs)),
    }
    empty_terms = {field: [] for field in _TERM_FIELDS}
    treated = analysis.pairs["treated_pos"].to_numpy(dtype=int)
    control = analysis.pairs["control_pos"].to_numpy(dtype=int)
    member_texts = [str(texts_train[pos]) for pos in treated] + [
        str(texts_train[pos]) for pos in control
    ]
    try:
        vectorizer = make_bow_vectorizer(dict(group["vectorizer_params"]))
        members = sparse.csr_matrix(vectorizer.fit_transform(member_texts), dtype=float)
    except ValueError as exc:
        view.update(
            {"status": "empty_vocabulary", "reason": str(exc), "n_features": 0,
             "n_nonzero_terms": 0, **empty_terms}
        )
        return {"view": view, "test_tau": None}
    n = len(treated)
    pair_features = 0.5 * (members[:n] + members[n:])
    mean = np.asarray(pair_features.mean(axis=0)).ravel()
    square = np.asarray(pair_features.multiply(pair_features).mean(axis=0)).ravel()
    sd = np.sqrt(np.maximum(square - mean * mean, 0.0))
    keep = np.flatnonzero(sd > _SD_FLOOR)
    terms = np.asarray(vectorizer.get_feature_names_out()).astype(str)
    view["n_features"] = int(len(keep))
    if not keep.size:
        view.update(
            {"status": "no_varying_features",
             "reason": "no pair feature varies across the analysis pairs",
             "n_nonzero_terms": 0, **empty_terms}
        )
        return {"view": view, "test_tau": None}
    scale = 1.0 / sd[keep]
    z = sparse.csr_matrix(pair_features[:, keep].multiply(scale[None, :]))
    means = mean[keep] * scale
    fitted = fit_penalized_pair_model(
        z,
        means,
        analysis.target,
        analysis.offset,
        loss=analysis.loss,
        l1_ratio=settings.l1_ratio,
        cv_folds=settings.cv_folds,
        max_iter=settings.max_iter,
        seed=seed,
    )
    standardized = np.asarray(fitted["beta"], dtype=float)
    raw = standardized * scale
    raw_intercept = float(fitted["intercept"] - float(means @ standardized))
    positive, negative = _term_rows(terms[keep], standardized, raw, settings.top_n)
    view.update(
        {
            "status": "fitted",
            "reason": None,
            "n_nonzero_terms": int(np.count_nonzero(standardized)),
            # The intercept of the centered fit: the effect at the mean pair
            # features (average conditional log odds ratio, or average effect).
            "average_effect": float(fitted["intercept"]),
            "intercept": raw_intercept,
            "penalty": {
                "l1_ratio": float(settings.l1_ratio),
                "selection": "minimum_cross_validated_heldout_loss",
                "selected_lambda": fitted["selected_lambda"],
                "selected_index": fitted["selected_index"],
                "lambda_max": fitted["lambda_max"],
                "lambda_min": fitted["lambda_min"],
                "grid_size": fitted["penalty_grid_size"],
                "cv_folds": fitted["cv_folds"],
                "cv_mean_heldout_loss": fitted["cv_mean_heldout_loss"],
            },
            "constant_effect": float(fitted["constant_intercept"]),
            "validation": fitted["validation"],
            "solver": {**fitted["solver"], "max_iter": int(settings.max_iter)},
            "effect_modification_positive": positive,
            "effect_modification_negative": negative,
        }
    )
    test_tau = np.full(len(texts_test), raw_intercept, dtype=float)
    if len(texts_test):
        test_features = sparse.csr_matrix(
            vectorizer.transform([str(text) for text in texts_test]), dtype=float
        )[:, keep]
        test_tau = raw_intercept + np.asarray(test_features @ raw).ravel()
    return {"view": view, "test_tau": test_tau}


def _validate_view_payload(payload: Mapping[str, Any], *, test_rows: int) -> None:
    if not isinstance(payload.get("view"), Mapping):
        raise ValueError("matched-pair view checkpoint is invalid")
    tau = payload.get("test_tau")
    if tau is not None and len(np.asarray(tau)) != int(test_rows):
        raise ValueError("matched-pair view checkpoint test row count changed")


# ---------------------------------------------------------------------------
# Continuous-outcome prognostic score
# ---------------------------------------------------------------------------


def crossfit_prognostic_score(
    texts: Sequence[str],
    outcome: np.ndarray,
    treatment: np.ndarray,
    *,
    vectorizer_params: Mapping[str, Any],
    folds: int,
    l1_ratio: float,
    cv_folds: int,
    max_iter: int,
    seed: int,
) -> Dict[str, Any]:
    """Cross-fitted untreated-arm outcome mean mu0 for every training row.

    In each fold the vectorizer is fitted on the fold's fitting rows and an
    elastic-net linear regression (standardized columns, penalty chosen by
    K-fold CV) on the fold's untreated fitting rows; the fold's held-out rows
    are predicted. No row's prediction comes from a model trained on it.
    """

    y = np.asarray(outcome, dtype=float)
    t = np.asarray(treatment, dtype=float)
    n = len(y)
    if n < 2:
        raise ValueError("prognostic cross-fitting requires at least two rows")
    n_folds = max(2, min(int(folds), n))
    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=int(seed))
    prognostic = np.full(n, np.nan, dtype=float)
    fold_audit: List[Dict[str, Any]] = []
    for fold, (fit, heldout) in enumerate(splitter.split(np.zeros(n)), start=1):
        fit = np.asarray(fit, dtype=int)
        heldout = np.asarray(heldout, dtype=int)
        controls = fit[t[fit] == 0]
        record: Dict[str, Any] = {"fold": fold, "n_untreated_fit_rows": int(len(controls))}
        if len(controls) < 3 or float(np.var(y[controls])) <= 1e-15:
            value = float(np.mean(y[controls])) if len(controls) else float(np.mean(y[fit]))
            prognostic[heldout] = value
            record["constant_reason"] = "too_few_untreated_rows" if len(controls) < 3 else "constant_target"
            fold_audit.append(record)
            continue
        vectorizer = make_bow_vectorizer(dict(vectorizer_params))
        try:
            x_fit = vectorizer.fit_transform([str(texts[int(pos)]) for pos in fit])
        except ValueError:
            prognostic[heldout] = float(np.mean(y[controls]))
            record["constant_reason"] = "empty_vocabulary"
            fold_audit.append(record)
            continue
        position = {int(pos): index for index, pos in enumerate(fit)}
        x_controls = x_fit[[position[int(pos)] for pos in controls]]
        x_heldout = vectorizer.transform([str(texts[int(pos)]) for pos in heldout])
        scaler = StandardScaler(with_mean=False).fit(x_controls)
        inner = KFold(
            n_splits=max(2, min(int(cv_folds), len(controls))),
            shuffle=True,
            random_state=int(seed) + fold,
        )
        model = ElasticNetCV(
            l1_ratio=float(l1_ratio),
            n_alphas=PENALTY_GRID_SIZE,
            cv=inner,
            fit_intercept=True,
            max_iter=int(max_iter),
            selection="cyclic",
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(scaler.transform(x_controls), y[controls])
        prognostic[heldout] = model.predict(scaler.transform(x_heldout))
        record.update(
            {
                "selected_alpha": float(model.alpha_),
                "n_nonzero_coefficients": int(np.count_nonzero(model.coef_)),
                "convergence_warnings": int(
                    sum(isinstance(item.message, ConvergenceWarning) for item in caught)
                ),
            }
        )
        fold_audit.append(record)
    return {
        "prognostic": prognostic,
        "audit": {
            "source": CONTINUOUS_PROGNOSTIC_SOURCE,
            "model": "elastic-net linear regression on untreated fitting rows",
            "folds": int(n_folds),
            "l1_ratio": float(l1_ratio),
            "cv_folds": int(cv_folds),
            "fold_fits": fold_audit,
        },
    }


# ---------------------------------------------------------------------------
# Lane orchestration
# ---------------------------------------------------------------------------


@dataclass
class MatchedPairLaneResult:
    importance: Dict[str, Any]
    metrics: Dict[str, Any]
    prediction_frames: List[pd.DataFrame]


CheckpointRunner = Callable[..., Any]


def _direct_checkpoint(unit: str, compute: Callable[[], Any], **_kwargs: Any) -> Any:
    del unit
    return compute()


def run_matched_pair_lane(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    texts_train: Sequence[str],
    texts_test: Sequence[str],
    y_train: np.ndarray,
    t_train: np.ndarray,
    propensity: np.ndarray,
    prognostic: np.ndarray,
    score_sources: Mapping[str, Any],
    vectorizer_groups: Sequence[Mapping[str, Any]],
    settings: MatchedPairSettings,
    outer_fold: int,
    seed: int,
    checkpoint: Optional[CheckpointRunner] = None,
) -> MatchedPairLaneResult:
    """Match, fit one pair model per vectorizer, and assemble the evidence.

    ``propensity`` and ``prognostic`` must be out of fold for every training
    row. Below the minimum pair count the lane emits no evidence for the
    context and records why.
    """

    checkpoint = checkpoint or _direct_checkpoint
    y = np.asarray(y_train, dtype=float)
    t = np.asarray(t_train, dtype=float)
    row_ids = (
        train_df["_oci_row_id"].to_numpy() if "_oci_row_id" in train_df.columns
        else np.arange(len(train_df))
    )
    caliper = resolve_prognostic_caliper(settings, np.asarray(prognostic, dtype=float))
    pairs, matching = build_matched_pairs(
        row_ids=row_ids,
        treatment=t,
        propensity=propensity,
        prognostic=prognostic,
        propensity_caliper=settings.propensity_caliper,
        prognostic_caliper=caliper["absolute"],
    )
    matching.update(summarize_pairs(pairs, y, binary=settings.binary))
    matching["prognostic_caliper"] = caliper
    analysis = pair_analysis_data(pairs, y, binary=settings.binary)
    n_analysis = int(len(analysis.pairs))
    required = settings.min_discordant_pairs if settings.binary else settings.min_pairs
    # Nested pair cross-fitting needs at least two pairs per fold.
    required = max(int(required), 4)
    status, reason = "fitted", None
    if n_analysis < required:
        status = "insufficient_discordant_pairs" if settings.binary else "insufficient_pairs"
        noun = "discordant pairs" if settings.binary else "matched pairs"
        key = "matched_pair_min_discordant_pairs" if settings.binary else "matched_pair_min_pairs"
        reason = (
            f"{n_analysis} {noun}; the lane requires at least {required} "
            f"({key}) and emits no evidence for this context"
        )
    views: List[Dict[str, Any]] = []
    prediction_frames: List[pd.DataFrame] = []
    if status == "fitted":
        analysis_identity = lightweight_fingerprint(
            {
                "treated": analysis.pairs["treated_row_id"].tolist(),
                "control": analysis.pairs["control_row_id"].tolist(),
                "target": np.round(analysis.target, 12).tolist(),
                "offset": np.round(analysis.offset, 12).tolist(),
            }
        )
        for group in vectorizer_groups:
            group_index = int(group["group_index"])
            view_seed = int((int(seed) + 1_009 * group_index) % (2**31 - 1))

            def compute(group: Mapping[str, Any] = group, view_seed: int = view_seed) -> Dict[str, Any]:
                return fit_matched_pair_view(
                    analysis=analysis,
                    texts_train=texts_train,
                    texts_test=texts_test,
                    group=group,
                    settings=settings,
                    seed=view_seed,
                )

            payload = checkpoint(
                f"bow/matched_pair/vectorizer_{group_index:03d}",
                compute,
                parameters={
                    "checkpoint_version": MATCHED_PAIR_CHECKPOINT_VERSION,
                    "outer_fold": int(outer_fold),
                    "settings": settings.identity(),
                    "vectorizer": dict(group["vectorizer_params"]),
                    "view_names": list(group["view_names"]),
                    "analysis_pairs": analysis_identity,
                    "seed": view_seed,
                },
                validator=lambda value: _validate_view_payload(value, test_rows=len(test_df)),
            )
            view = dict(payload["view"])
            views.append(view)
            tau = payload.get("test_tau")
            if view.get("status") == "fitted" and tau is not None:
                prediction_frames.append(
                    pd.DataFrame(
                        {
                            "_oci_row_id": test_df["_oci_row_id"].to_numpy(),
                            "outer_fold": int(outer_fold),
                            "split_role": "test_outer_train_fit",
                            "source_name": f"bow__{group['name']}__matched_pair",
                            "tau_hat_matched_pair": np.asarray(tau, dtype=float),
                        }
                    )
                )
    importance = {
        "schema_version": MATCHED_PAIR_EVIDENCE_SCHEMA_VERSION,
        "outcome_type": settings.outcome_type,
        "method": (
            "conditional_logistic_regression_on_discordant_pairs"
            if settings.binary
            else "elastic_net_regression_of_adjusted_paired_differences"
        ),
        "effect_scale": (
            "conditional_log_odds_ratio" if settings.binary else "outcome_difference"
        ),
        "pair_features": "mean_of_member_bow_vectors",
        "score_sources": dict(score_sources),
        "status": status,
        "reason": reason,
        "minimum_analysis_pairs": int(required),
        "n_analysis_pairs": n_analysis,
        "matching": matching,
        "n_views": len(views),
        "views": views,
    }
    metrics: Dict[str, Any] = {
        "matched_pair_status": status,
        "matched_pair_n_pairs": int(matching.get("n_pairs") or 0),
        "matched_pair_n_analysis_pairs": n_analysis,
    }
    for view in views:
        validation = view.get("validation") or {}
        gain = validation.get(
            "heldout_log_likelihood_gain_per_pair"
            if settings.binary
            else "heldout_mse_reduction"
        )
        metrics[f"{view['view_name']}_heldout_gain"] = _finite_or_none(gain)
        metrics[f"{view['view_name']}_n_nonzero_terms"] = int(view.get("n_nonzero_terms") or 0)
    if status != "fitted":
        logger.info("Outer fold %s matched_pair_uplift: %s", outer_fold, reason)
    return MatchedPairLaneResult(
        importance=importance,
        metrics=metrics,
        prediction_frames=prediction_frames,
    )


__all__ = [
    "BINARY_PROGNOSTIC_SOURCE",
    "CONTINUOUS_PROGNOSTIC_SOURCE",
    "MATCHED_PAIR_CHECKPOINT_VERSION",
    "MAX_CANDIDATE_CONTROLS",
    "MatchedPairLaneResult",
    "MatchedPairSettings",
    "PROPENSITY_SOURCE",
    "PairAnalysisData",
    "build_matched_pairs",
    "constant_intercept",
    "crossfit_prognostic_score",
    "distinct_vectorizer_groups",
    "elastic_net_path",
    "fit_matched_pair_view",
    "fit_penalized_pair_model",
    "hopcroft_karp",
    "matched_pair_seed",
    "pair_analysis_data",
    "penalty_grid",
    "probability_logit",
    "resolve_prognostic_caliper",
    "run_matched_pair_lane",
    "summarize_pairs",
]
