"""Standalone dispatch for the retained explicit-feature workflows.

The production multi-model path is owned by
``ResearchAllEvidenceWorkflow``.  This module intentionally contains no
fallback neural head: every supported standalone model has an explicit,
auditable dispatch branch.

It also owns the policies shared by those runners: the seeded
treatment x outcome stratified outer splits, the single oracle-name rule
(oracle columns are withheld until predictions are frozen), the pooled
cross-fitted doubly robust ATE summary, and post-hoc oracle evaluation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

from ..config import AppliedInferenceConfig

logger = logging.getLogger(__name__)

# Synthetic-oracle columns (``true_*`` from the DGP, ``oracle_*``,
# ``ground_truth*``). This is the single oracle-name rule for the standalone
# runners: such columns are removed before any runner sees the data and are
# joined back only after predictions are frozen on disk.
ORACLE_NAME_PREFIXES = ("true_", "oracle_", "ground_truth")
ROW_ID_COLUMN = "_oci_row_id"

STANDALONE_MODEL_TYPES = (
    "explicit_feature_forest",
    "agentic_explicit_feature_forest",
    "agentic_attention_variable_forest",
    "multi_model_agentic_forest",
)


def is_oracle_name(name: Any) -> bool:
    """Return True for oracle column or metric names (one shared rule)."""

    return str(name).lower().startswith(ORACLE_NAME_PREFIXES)


def drop_oracle_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return ``frame`` without oracle columns."""

    return frame.drop(columns=[c for c in frame.columns if is_oracle_name(c)])


def without_oracle_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``metrics`` without oracle-named keys."""

    return {key: value for key, value in metrics.items() if not is_oracle_name(key)}


def resolve_oracle_ite_column(columns: Sequence[Any], outcome_type: str) -> Optional[str]:
    """Pick the oracle ITE column on the analysis outcome scale.

    ``true_ite`` is on the outcome scale for both outcome types. Legacy
    datasets only have ``true_ite_prob`` (probability scale), which is the
    outcome scale only for binary outcomes.
    """

    names = {str(column) for column in columns}
    if "true_ite" in names:
        return "true_ite"
    if "true_ite_prob" in names and str(outcome_type).lower() == "binary":
        return "true_ite_prob"
    return None


def required_applied_seed(config: AppliedInferenceConfig) -> int:
    """Return the configured scientific seed or fail closed."""

    seed = getattr(config, "seed", None)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31:
        raise ValueError(
            "applied inference requires an explicit integer seed in [0, 2**31); "
            "set ExperimentConfig.seed (oci run copies it to applied_inference.seed) "
            "or AppliedInferenceConfig.seed for direct calls"
        )
    return int(seed)


def stratified_fold_indices(
    frame: pd.DataFrame,
    *,
    treatment_column: str,
    outcome_column: str,
    outcome_type: str,
    n_splits: int,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Seeded StratifiedKFold on treatment x outcome (one rule for all runners).

    Continuous outcomes are stratified on treatment x (outcome above its
    median). If a treatment x outcome stratum is smaller than ``n_splits`` the
    strata coarsen to treatment alone; if a treatment arm is smaller than
    ``n_splits`` this fails instead of silently using unstratified folds.
    """

    n_splits = int(n_splits)
    if n_splits < 2:
        raise ValueError("stratified folds require n_splits >= 2")
    if n_splits > len(frame):
        raise ValueError(f"n_splits={n_splits} exceeds n={len(frame)}")
    treatment = frame[treatment_column].reset_index(drop=True).astype(str)
    outcome = frame[outcome_column].reset_index(drop=True)
    if str(outcome_type).lower() == "continuous":
        values = pd.to_numeric(outcome, errors="coerce")
        outcome_stratum = (values > values.median()).astype(int).astype(str)
    else:
        outcome_stratum = outcome.astype(str)
    strata = treatment + "_" + outcome_stratum
    if strata.value_counts().min() < n_splits:
        strata = treatment
        smallest = int(strata.value_counts().min())
        if smallest < n_splits:
            raise ValueError(
                f"cannot build {n_splits} stratified folds: the smallest treatment arm "
                f"has {smallest} row(s); reduce the number of folds or add rows"
            )
        logger.warning(
            "Treatment x outcome strata are smaller than %s rows; stratifying folds "
            "on treatment only",
            n_splits,
        )
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    return [
        (np.asarray(train_idx), np.asarray(test_idx))
        for train_idx, test_idx in splitter.split(np.zeros(len(frame)), strata.to_numpy())
    ]


def has_heldout_test_split(dataset: pd.DataFrame, config: AppliedInferenceConfig) -> bool:
    split_col = config.split_column
    return bool(split_col in dataset.columns and "test" in set(dataset[split_col]))


def standalone_outer_splits(
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    """Outer analysis splits shared by the standalone explicit-feature runners.

    ``cv_folds > 1`` gives seeded treatment x outcome stratified folds using the
    configured seed. Otherwise the ``split_column`` must hold a "test" split
    (train/val rows fit, test rows are held out). Without either, discovery and
    estimation would use the same rows; that fails closed unless
    ``explicit_feature_forest.allow_non_honest_outer_split`` is true.
    """

    if config.cv_folds > 1:
        folds = stratified_fold_indices(
            dataset,
            treatment_column=config.treatment_column,
            outcome_column=config.outcome_column,
            outcome_type=config.outcome_type,
            n_splits=config.cv_folds,
            seed=required_applied_seed(config),
        )
        return [
            (fold, train_idx, test_idx)
            for fold, (train_idx, test_idx) in enumerate(folds, start=1)
        ]

    if has_heldout_test_split(dataset, config):
        split = dataset[config.split_column]
        train_idx = np.where(split.isin(["train", "val"]).to_numpy())[0]
        test_idx = np.where((split == "test").to_numpy())[0]
        if len(train_idx) == 0:
            raise ValueError(
                f"split column {config.split_column!r} has a 'test' split but no "
                "'train'/'val' rows to fit on"
            )
        return [(1, train_idx, test_idx)]

    cf_config = getattr(config.architecture, "explicit_feature_forest", None)
    if not bool(getattr(cf_config, "allow_non_honest_outer_split", False)):
        raise ValueError(
            "No held-out outer split: set cv_folds > 1, or provide a "
            f"{config.split_column!r} column with 'test' rows. To deliberately run "
            "discovery and estimation on the same rows (non-honest), set "
            "architecture.explicit_feature_forest.allow_non_honest_outer_split=true."
        )
    logger.warning(
        "No held-out split configured; allow_non_honest_outer_split=true, so variable "
        "discovery and final estimates use the full dataset (non-honest)."
    )
    all_idx = np.arange(len(dataset))
    return [(1, all_idx, all_idx)]


def _run_explicit_feature_extraction(
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    output_path: Path,
) -> Tuple[pd.DataFrame, List[str]]:
    """Materialize investigator-specified explicit features, with caching.

    Uses the shared explicit-feature extraction provider, so endpoint, retry,
    reasoning, temporal, grouping, and context-strategy settings are honored
    exactly as in the agentic explicit-feature runners.
    """

    from .agentic_explicit_feature_forest import VLLMExplicitFeatureExtractionProvider

    specs = list(config.explicit_features.features)
    if not specs:
        raise ValueError(
            "explicit_feature_forest requires at least one explicit feature contract"
        )
    provider = VLLMExplicitFeatureExtractionProvider(
        config=config,
        output_dir=Path(output_path).parent / "explicit_feature_extraction",
    )
    enriched = provider.ensure_features(dataset, specs)
    return enriched, [f"explicit_feat_{spec.name}" for spec in specs]


def run_applied_inference(
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    output_path: Path,
    device: torch.device,
    gpu_ids: Optional[List[int]] = None,
    num_workers: int = 1,
) -> None:
    """Dispatch one retained standalone explicit-feature architecture.

    Oracle columns are removed once here, before any runner, extractor or agent
    can see them. After the runner has written its predictions, the frozen
    predictions file is hashed and only then joined to the oracle columns for
    post-hoc evaluation.
    """

    model_type = str(config.architecture.model_type)
    if model_type == "multi_model_forest":
        raise RuntimeError(
            "model_type='multi_model_forest' must run through "
            "ResearchAllEvidenceWorkflow (scripts/run_all_evidence.py)"
        )
    if model_type not in STANDALONE_MODEL_TYPES:
        raise ValueError(
            f"Unsupported retired model_type={model_type!r}; retained standalone types: "
            + ", ".join(STANDALONE_MODEL_TYPES)
        )

    dataset = dataset.reset_index(drop=True)
    oracle_columns = [column for column in dataset.columns if is_oracle_name(column)]
    protected = {config.text_column, config.treatment_column, config.outcome_column}
    clashing = sorted(protected.intersection(oracle_columns))
    if clashing:
        raise ValueError(
            f"columns {clashing} are configured as model inputs but use an oracle "
            f"name prefix {ORACLE_NAME_PREFIXES}; rename them"
        )
    row_ids = np.arange(len(dataset), dtype=int)
    oracle = dataset[oracle_columns].copy()
    oracle[ROW_ID_COLUMN] = row_ids
    dataset = dataset.drop(columns=oracle_columns)
    dataset[ROW_ID_COLUMN] = row_ids
    if oracle_columns:
        logger.info(
            "Withholding %s oracle column(s) from the standalone runner until "
            "predictions are frozen",
            len(oracle_columns),
        )
    # Fail closed on a missing seed or a non-honest outer split before any
    # LLM extraction or model training starts.
    required_applied_seed(config)
    if model_type != "agentic_explicit_feature_forest":
        standalone_outer_splits(dataset, config)

    _dispatch_standalone(
        model_type=model_type,
        dataset=dataset,
        config=config,
        output_path=output_path,
        device=device,
        gpu_ids=gpu_ids,
        num_workers=num_workers,
    )

    output_path = Path(output_path)
    honest = (
        model_type == "agentic_explicit_feature_forest"
        or config.cv_folds > 1
        or has_heldout_test_split(dataset, config)
    )
    write_outer_ate_summary(output_path, honest=honest)
    if oracle_columns:
        write_posthoc_oracle_metrics(output_path, oracle, outcome_type=config.outcome_type)


def write_outer_ate_summary(predictions_path: Path, *, honest: bool) -> Optional[Path]:
    """Pool per-row held-out doubly robust scores into one cross-fitted ATE."""

    from ..models.causal_forest_head import summarize_doubly_robust_scores

    predictions_path = Path(predictions_path)
    if not predictions_path.exists():
        return None
    predictions = pd.read_parquet(predictions_path)
    if "pred_dr_score" not in predictions.columns:
        return None
    summary = summarize_doubly_robust_scores(predictions["pred_dr_score"].to_numpy())
    payload = {
        "schema_version": "standalone_dr_ate_v1",
        "estimator": "cross_fitted_aipw",
        "description": (
            "Mean of per-row AIPW scores on outer held-out rows; each score uses the "
            "outer-fold causal forest tau(X) and its out-of-sample elastic-net "
            "nuisances, with e(1-e) floored at 0.01."
        ),
        "honest_outer_holdout": bool(honest),
        **summary,
        "mean_predicted_cate": float(predictions["pred_ite_prob"].mean())
        if "pred_ite_prob" in predictions.columns
        else None,
    }
    if not honest:
        payload["warning"] = (
            "non-honest run: scores were computed on the rows used for discovery "
            "and fitting, so the standard error is not valid"
        )
    path = predictions_path.parent / "ate_summary.json"
    path.write_text(json.dumps(payload, indent=2))
    logger.info(
        "Cross-fitted doubly robust ATE: %s (SE %s, n=%s)",
        payload["ate"],
        payload["ate_stderr"],
        payload["n"],
    )
    return path


def _finite_pair(left: np.ndarray, right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(left) & np.isfinite(right)
    return left[mask], right[mask]


def _oracle_ite_metrics(truth: np.ndarray, predicted: np.ndarray) -> Dict[str, Any]:
    truth, predicted = _finite_pair(
        np.asarray(truth, dtype=float), np.asarray(predicted, dtype=float)
    )
    n = int(truth.size)
    metrics: Dict[str, Any] = {"n": n}
    if n == 0:
        return metrics
    errors = predicted - truth
    metrics.update(
        {
            "ite_mse": float(np.mean(errors**2)),
            "ite_mae": float(np.mean(np.abs(errors))),
            "mean_predicted_ite": float(np.mean(predicted)),
            "true_ate": float(np.mean(truth)),
            "mean_predicted_ite_bias": float(np.mean(predicted) - np.mean(truth)),
            "ite_pearson": None,
            "ite_spearman": None,
        }
    )
    if n > 1 and np.std(truth) > 0 and np.std(predicted) > 0:
        metrics["ite_pearson"] = float(np.corrcoef(truth, predicted)[0, 1])
        metrics["ite_spearman"] = float(
            pd.Series(truth).rank().corr(pd.Series(predicted).rank())
        )
    return metrics


def write_posthoc_oracle_metrics(
    predictions_path: Path,
    oracle: pd.DataFrame,
    *,
    outcome_type: str,
) -> Path:
    """Join oracle truth to frozen predictions and write post-hoc metrics."""

    predictions_path = Path(predictions_path)
    output = predictions_path.parent / "posthoc_oracle_metrics.json"
    payload: Dict[str, Any] = {
        "schema_version": "standalone_posthoc_oracle_metrics_v1",
        "predictions_frozen_before_oracle_join": True,
        "oracle_columns_consumed_by_modeling": False,
        "oracle_columns": sorted(c for c in oracle.columns if c != ROW_ID_COLUMN),
    }
    if not predictions_path.exists():
        payload.update(status="unavailable", reason="predictions file was not written")
        output.write_text(json.dumps(payload, indent=2))
        return output
    payload["predictions_sha256"] = hashlib.sha256(predictions_path.read_bytes()).hexdigest()
    predictions = pd.read_parquet(predictions_path)
    leaked = [column for column in predictions.columns if is_oracle_name(column)]
    if leaked:
        raise RuntimeError(f"frozen standalone predictions contain oracle columns: {leaked}")
    ite_column = resolve_oracle_ite_column(oracle.columns, outcome_type)
    payload["oracle_ite_column"] = ite_column
    if ROW_ID_COLUMN not in predictions.columns or "pred_ite_prob" not in predictions.columns:
        payload.update(
            status="unavailable",
            reason=f"predictions lack {ROW_ID_COLUMN!r} or 'pred_ite_prob'",
        )
        output.write_text(json.dumps(payload, indent=2))
        return output
    joined = predictions.merge(oracle, on=ROW_ID_COLUMN, how="left", validate="many_to_one")
    metrics: Dict[str, Any] = {}
    if ite_column is None:
        payload["ite_status"] = (
            "unavailable: no outcome-scale oracle ITE column (a continuous-outcome "
            "dataset with only probability-scale true_ite_prob is not comparable)"
        )
    else:
        truth = joined[ite_column].to_numpy(dtype=float)
        metrics.update(_oracle_ite_metrics(truth, joined["pred_ite_prob"].to_numpy(dtype=float)))
        if "pred_dr_score" in joined.columns:
            scores, scored_truth = _finite_pair(
                joined["pred_dr_score"].to_numpy(dtype=float), truth
            )
            if scores.size:
                metrics["dr_ate"] = float(np.mean(scores))
                metrics["dr_ate_bias"] = float(np.mean(scores) - np.mean(scored_truth))
                if scores.size > 1:
                    metrics["dr_ate_stderr"] = float(
                        np.std(scores, ddof=1) / math.sqrt(scores.size)
                    )
        if "outer_fold" in joined.columns:
            payload["by_outer_fold"] = [
                {
                    "outer_fold": int(fold),
                    **_oracle_ite_metrics(
                        group[ite_column].to_numpy(dtype=float),
                        group["pred_ite_prob"].to_numpy(dtype=float),
                    ),
                }
                for fold, group in joined.groupby("outer_fold", sort=True)
            ]
    if {"true_treatment_prob", "pred_propensity_prob"}.issubset(joined.columns):
        truth, predicted = _finite_pair(
            joined["true_treatment_prob"].to_numpy(dtype=float),
            joined["pred_propensity_prob"].to_numpy(dtype=float),
        )
        if truth.size > 1 and np.std(truth) > 0 and np.std(predicted) > 0:
            metrics["propensity_pearson"] = float(np.corrcoef(truth, predicted)[0, 1])
            metrics["propensity_mae"] = float(np.mean(np.abs(truth - predicted)))
    payload["status"] = "complete"
    payload["metrics"] = metrics
    output.write_text(json.dumps(payload, indent=2))
    logger.info("Post-hoc oracle metrics saved to: %s", output)
    return output


def _dispatch_standalone(
    *,
    model_type: str,
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    output_path: Path,
    device: torch.device,
    gpu_ids: Optional[List[int]],
    num_workers: int,
) -> None:
    if model_type == "agentic_explicit_feature_forest":
        from .agentic_explicit_feature_forest import run_agentic_explicit_feature_forest

        run_agentic_explicit_feature_forest(
            dataset=dataset,
            config=config,
            output_path=output_path,
            device=device,
            num_workers=num_workers,
        )
        return

    if model_type == "agentic_attention_variable_forest":
        from .agentic_attention_variable_forest import run_agentic_attention_variable_forest

        run_agentic_attention_variable_forest(
            dataset=dataset,
            config=config,
            output_path=output_path,
            device=device,
            num_workers=num_workers,
            gpu_ids=gpu_ids,
        )
        return

    if model_type == "multi_model_agentic_forest":
        from .multi_model_agentic_forest import run_multi_model_agentic_forest

        run_multi_model_agentic_forest(
            dataset=dataset,
            config=config,
            output_path=output_path,
            device=device,
            gpu_ids=gpu_ids,
            num_workers=num_workers,
        )
        return

    if model_type == "explicit_feature_forest":
        from .applied_explicit_feature_forest import (
            run_applied_inference_explicit_feature_forest,
        )

        explicit_columns = None
        if config.explicit_features.enabled:
            dataset, explicit_columns = _run_explicit_feature_extraction(
                dataset,
                config,
                output_path,
            )
        run_applied_inference_explicit_feature_forest(
            dataset=dataset,
            config=config,
            output_path=output_path,
            device=device,
            num_workers=num_workers,
            explicit_feature_columns=explicit_columns,
        )
        return

    raise ValueError(f"Unsupported standalone model_type={model_type!r}")


__all__ = [
    "ORACLE_NAME_PREFIXES",
    "STANDALONE_MODEL_TYPES",
    "drop_oracle_columns",
    "is_oracle_name",
    "required_applied_seed",
    "resolve_oracle_ite_column",
    "run_applied_inference",
    "standalone_outer_splits",
    "stratified_fold_indices",
    "without_oracle_metrics",
    "write_outer_ate_summary",
    "write_posthoc_oracle_metrics",
]
