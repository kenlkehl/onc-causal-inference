"""Small modeling helpers shared by active Stage 1 lanes.

These functions used to live inside the retired all-in-one agentic runner. The
research workflow depends on the computations, not that orchestration surface,
so they live here without importing extraction agents or model runtimes.
"""

from __future__ import annotations

import unicodedata
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, mean_squared_error

from ..config import BoWViewConfig
from .causal_modeling_support import safe_roc_auc as _safe_roc_auc

_DASH_TRANSLATION = dict.fromkeys(
    map(ord, "\u2010\u2011\u2012\u2013\u2014\u2212"),
    "-",
)


def _finite_or_none(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).translate(_DASH_TRANSLATION)
    text = text.replace("\u2265", ">=").replace("\u2264", "<=")
    return text.lower()


def _normalize_texts(values: Sequence[Any]) -> List[str]:
    return [_normalize_text(value) for value in values]


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
    }
    htr_evidence = result.get("htr_evidence") or {}
    if htr_evidence:
        row["htr_evidence"] = htr_evidence
    if inner_fold is not None:
        row["inner_fold"] = int(inner_fold)
    if heldout_rows is not None:
        row["heldout_rows"] = int(heldout_rows)
    return row


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
        frame.copy(), on="_oci_row_id", how="left", sort=False
    )
    if len(aligned) != len(discovery_df):
        raise ValueError(f"{source} predictions could not be aligned to discovery rows")
    for column in required_columns:
        values = pd.to_numeric(aligned[column], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{source} predictions contain non-finite {column} values")
        aligned[column] = values
    return aligned


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
        treatment = discovery_df[treatment_column].to_numpy(dtype=float)
        metrics.update(
            {
                "treatment_auroc": _safe_roc_auc(treatment, e_hat),
                "treatment_brier": _safe_brier_score(treatment, e_hat),
                "treatment_log_loss": _safe_log_loss(treatment, e_hat),
            }
        )
    if outcome_column in discovery_df.columns:
        outcome = discovery_df[outcome_column].to_numpy(dtype=float)
        if str(outcome_type).lower() == "continuous":
            metrics["outcome_rmse"] = _finite_or_none(np.sqrt(mean_squared_error(outcome, m_hat)))
        else:
            metrics.update(
                {
                    "outcome_auroc": _safe_roc_auc(outcome, m_hat),
                    "outcome_brier": _safe_brier_score(outcome, m_hat),
                    "outcome_log_loss": _safe_log_loss(outcome, m_hat),
                }
            )
    for column in ("y_residual", "t_residual", "r_pseudo_outcome"):
        if column not in predictions.columns:
            continue
        values = pd.to_numeric(predictions[column], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        metrics[f"{column}_mean"] = _finite_or_none(np.mean(finite)) if len(finite) else None
        metrics[f"{column}_std"] = _finite_or_none(np.std(finite)) if len(finite) else None
    return metrics


def _htr_effect_metrics(predictions: pd.DataFrame) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {"n_rows": int(len(predictions))}
    for column in (
        "tau_hat_r_stage",
        "tau_logit_modifier",
        "r_pseudo_outcome",
        "r_loss",
        "effect_loss",
        "effect_loss_at_zero_tau",
    ):
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
        sources = sorted(
            {str(value) for value in predictions["target_source"].dropna() if str(value)}
        )
        if sources:
            metrics["target_sources"] = sources
    return metrics


def _top_phrase_feature_rows(
    features: np.ndarray,
    *,
    top_n: int,
    treatment_coef: np.ndarray,
    outcome_coef: np.ndarray,
    pseudo_target_coef: np.ndarray,
    confounder_score: np.ndarray,
) -> List[Dict[str, Any]]:
    """Return phrase-biased two-to-four-token evidence rows."""

    phrase_indices = [
        index
        for index, feature in enumerate(features)
        if 2 <= len([token for token in str(feature).split() if token]) <= 4
    ]
    if not phrase_indices:
        return []

    def scale(values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        maximum = float(np.nanmax(np.abs(array))) if len(array) else 0.0
        if not np.isfinite(maximum) or maximum <= 0.0:
            return np.zeros_like(array, dtype=float)
        return np.abs(array) / maximum

    treatment_abs = np.abs(treatment_coef)
    outcome_abs = np.abs(outcome_coef)
    pseudo_abs = np.abs(pseudo_target_coef)
    combined = np.maximum.reduce(
        [scale(treatment_abs), scale(outcome_abs), scale(pseudo_abs), scale(confounder_score)]
    )
    indices = np.asarray(phrase_indices, dtype=int)
    order = indices[np.argsort(combined[indices])[::-1]]
    return [
        {
            "feature": str(features[index]),
            "token_count": len(str(features[index]).split()),
            "combined_score": _finite_or_none(combined[index]),
            "confounder_overlap_score": _finite_or_none(confounder_score[index]),
            "treatment_score": _finite_or_none(treatment_coef[index]),
            "abs_treatment_score": _finite_or_none(treatment_abs[index]),
            "outcome_score": _finite_or_none(outcome_coef[index]),
            "abs_outcome_score": _finite_or_none(outcome_abs[index]),
            "pseudo_target_score": _finite_or_none(pseudo_target_coef[index]),
            "abs_pseudo_target_score": _finite_or_none(pseudo_abs[index]),
        }
        for index in order[:top_n]
    ]


def _agent_visible_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    def is_oracle_name(key: Any) -> bool:
        name = str(key).lower()
        return name.startswith("oracle_") or name.startswith("true_") or "true_" in name

    return {key: value for key, value in metrics.items() if not is_oracle_name(key)}


def _bow_view_to_dict(view: BoWViewConfig) -> Dict[str, Any]:
    if type(view) is not BoWViewConfig:
        raise TypeError("BoW view serialization requires BoWViewConfig")
    return asdict(view)


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
            entry = accumulator.setdefault(
                _normalize_text(feature),
                {
                    "feature": feature,
                    "supporting_views": set(),
                    "view_scores": [],
                    "abs_confounder_scores": [],
                    "abs_effect_scores": [],
                },
            )
            entry["supporting_views"].add(view_name)
            entry["abs_confounder_scores"].append(
                abs(float(row.get("confounder_overlap_score") or 0.0))
            )
            entry["abs_effect_scores"].append(abs(float(row.get("abs_pseudo_target_score") or 0.0)))
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
        confounder = entry["abs_confounder_scores"]
        effect = entry["abs_effect_scores"]
        supporting = sorted(entry["supporting_views"])
        rows.append(
            {
                "feature": entry["feature"],
                "supporting_view_count": len(supporting),
                "supporting_views": supporting,
                "best_abs_confounder_score": _finite_or_none(max(confounder, default=0.0)),
                "mean_abs_confounder_score": _finite_or_none(np.mean(confounder)),
                "best_abs_effect_score": _finite_or_none(max(effect, default=0.0)),
                "mean_abs_effect_score": _finite_or_none(np.mean(effect)),
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
    # Per-view "phrase_features" stay inside each view; the cross-view list is
    # published once, so downstream consumers do not count it twice.
    return {
        "n_views": len(views),
        "views": views,
        "phrase_consensus": consensus,
    }


__all__ = [
    "_agent_visible_metrics",
    "_agentic_discovery_handoff_row",
    "_align_htr_prediction_frame",
    "_htr_effect_metrics",
    "_htr_nuisance_metrics",
    "_multi_view_importance",
    "_normalize_texts",
    "_top_phrase_feature_rows",
]
