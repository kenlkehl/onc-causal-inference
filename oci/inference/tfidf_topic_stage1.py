"""Stage 1 orchestration for exact-scope TF-IDF topic discovery."""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
import logging
import multiprocessing as mp
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.model_selection import KFold, StratifiedKFold
from threadpoolctl import threadpool_limits

from ..config import AppliedInferenceConfig, MultiModelForestConfig
from .tfidf_topic_discovery import (
    DISCOVERY_SCHEMA_VERSION,
    HANDOFF_SCHEMA_VERSION,
    fit_tfidf_topic_context,
    row_set_fingerprint,
    stable_hash,
)
from .tfidf_topic_score_selection import ORPHAN_NGRAM_SCORE_SCHEMA_VERSION

logger = logging.getLogger(__name__)


TFIDF_TOPIC_SPLIT_SCHEMA_VERSION = "tfidf_topic_joint_treatment_outcome_v1"
TFIDF_TOPIC_DATASET_FINGERPRINT_VERSION = "tfidf_topic_model_inputs_v1"
_DEFAULT_STAGE1_SEED = 42
TFIDF_NESTED_CALIBRATION_SCHEMA_VERSION = "tfidf_nested_fit_calibration_v1"


def _resolve_tfidf_topic_stage1_parallel_backend(value: Any) -> Tuple[str, str]:
    """Return the canonical configured backend and its joblib backend name.

    ``processes`` intentionally retains the existing loky behavior.  The
    opt-in ``multiprocessing`` backend is accepted only when joblib will create
    a Linux fork pool, which avoids importing the entry-point module again in
    each worker.  Failing closed on spawn/forkserver keeps the option's runtime
    contract explicit and avoids silently reintroducing the re-import path.
    """
    backend = str(value).strip().lower()
    aliases = {
        "loky": "processes",
        "fork": "multiprocessing",
    }
    backend = aliases.get(backend, backend)
    if backend not in {"threads", "processes", "multiprocessing"}:
        raise ValueError(
            "multi_model_forest.outer_parallel_backend must be threads, "
            "processes, or multiprocessing"
        )
    if backend == "multiprocessing":
        start_method = mp.get_start_method(allow_none=True)
        if start_method is None:
            start_method = mp.get_context().get_start_method()
        if not sys.platform.startswith("linux") or start_method != "fork":
            raise ValueError(
                "multi_model_forest.outer_parallel_backend='multiprocessing' "
                "requires Linux with multiprocessing start method 'fork'; "
                f"platform={sys.platform!r} start_method={start_method!r}"
            )
    return (
        backend,
        {
            "threads": "threading",
            "processes": "loky",
            "multiprocessing": "multiprocessing",
        }[backend],
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")


def _configured_seed(config: AppliedInferenceConfig) -> int:
    """Return the run seed propagated onto the applied-inference config.

    ``AppliedInferenceConfig`` predates the top-level experiment seed, so older
    direct callers may not have attached it.  Keeping the historical default
    here preserves those callers while ensuring a propagated ``config.seed``
    controls every v2 outer and inner split.
    """
    configured = getattr(config, "seed", None)
    if configured is None:
        return _DEFAULT_STAGE1_SEED
    return int(configured)


def _tfidf_context_scope_seed(*, global_seed: int, scope_id: str) -> int:
    digest = hashlib.sha256(
        f"{int(global_seed)}\0{str(scope_id)}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def make_joint_treatment_outcome_splits(
    dataset: pd.DataFrame,
    *,
    treatment_column: str,
    outcome_column: str,
    outcome_type: str,
    n_splits: int,
    seed: int,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Dict[str, Any]]:
    """Build deterministic folds, stratifying jointly on treatment and outcome.

    Binary outcomes use their observed labels directly.  Continuous outcomes
    use deterministic quantile bins before constructing the joint strata.  A
    shuffled K-fold split is used only when at least one joint stratum is too
    small for the requested number of folds (or there is only one stratum).
    The returned metadata is suitable for persisted split provenance.
    """
    fold_count = int(n_splits)
    if fold_count < 2:
        raise ValueError("Joint treatment/outcome splitting requires n_splits >= 2")
    if len(dataset) < fold_count:
        raise ValueError(
            "Joint treatment/outcome splitting requires at least one row per fold; "
            f"rows={len(dataset)} folds={fold_count}"
        )
    missing = [
        column for column in (treatment_column, outcome_column) if column not in dataset.columns
    ]
    if missing:
        raise ValueError(f"Cannot construct joint strata; missing columns: {missing}")

    treatment = dataset[treatment_column].astype("string").fillna("<missing>")
    raw_outcome = dataset[outcome_column]
    outcome_semantics = "observed_categories"
    if str(outcome_type).strip().lower() == "continuous":
        numeric_outcome = pd.to_numeric(raw_outcome, errors="coerce")
        nonmissing_unique = int(numeric_outcome.nunique(dropna=True))
        quantile_count = min(5, max(2, fold_count), nonmissing_unique)
        if quantile_count >= 2:
            try:
                outcome = (
                    pd.qcut(
                        numeric_outcome,
                        q=quantile_count,
                        duplicates="drop",
                    )
                    .astype("string")
                    .fillna("<missing>")
                )
                outcome_semantics = f"quantile_bins_{int(outcome.nunique(dropna=False))}"
            except ValueError:
                outcome = raw_outcome.astype("string").fillna("<missing>")
        else:
            outcome = raw_outcome.astype("string").fillna("<missing>")
    else:
        outcome = raw_outcome.astype("string").fillna("<missing>")

    joint_labels = (
        treatment.str.replace("\x1f", " ", regex=False)
        + "\x1f"
        + outcome.str.replace("\x1f", " ", regex=False)
    ).to_numpy(dtype=str)
    counts = pd.Series(joint_labels, dtype="string").value_counts(dropna=False)
    minimum_count = int(counts.min()) if len(counts) else 0
    can_stratify = len(counts) >= 2 and minimum_count >= fold_count
    if can_stratify:
        method = "stratified_joint_treatment_outcome"
        fallback_reason = None
        iterator = StratifiedKFold(
            n_splits=fold_count,
            shuffle=True,
            random_state=int(seed),
        ).split(np.zeros(len(dataset), dtype=np.int8), joint_labels)
    else:
        method = "kfold_fallback"
        fallback_reason = (
            "fewer_than_two_joint_strata"
            if len(counts) < 2
            else "minimum_joint_stratum_count_below_requested_folds"
        )
        iterator = KFold(
            n_splits=fold_count,
            shuffle=True,
            random_state=int(seed),
        ).split(np.zeros(len(dataset), dtype=np.int8))
    splits = [
        (np.asarray(fit, dtype=int), np.asarray(heldout, dtype=int)) for fit, heldout in iterator
    ]
    metadata = {
        "schema_version": TFIDF_TOPIC_SPLIT_SCHEMA_VERSION,
        "method": method,
        "seed": int(seed),
        "n_splits": fold_count,
        "treatment_column": str(treatment_column),
        "outcome_column": str(outcome_column),
        "outcome_stratification": outcome_semantics,
        "n_joint_strata": int(len(counts)),
        "minimum_joint_stratum_count": minimum_count,
        "fallback_reason": fallback_reason,
    }
    return splits, metadata


def _outer_split_plan(
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Dict[str, Any]]:
    if int(config.cv_folds) > 1:
        return make_joint_treatment_outcome_splits(
            dataset,
            treatment_column=config.treatment_column,
            outcome_column=config.outcome_column,
            outcome_type=config.outcome_type,
            n_splits=int(config.cv_folds),
            seed=_configured_seed(config),
        )
    split_column = config.split_column
    if split_column in dataset.columns and "test" in set(dataset[split_column]):
        train = np.where(dataset[split_column].isin(["train", "val"]).to_numpy())[0]
        test = np.where((dataset[split_column] == "test").to_numpy())[0]
        if len(train) == 0 or len(test) == 0:
            raise ValueError("Explicit split must contain non-empty fit and test row sets")
        return [(train.astype(int), test.astype(int))], {
            "schema_version": TFIDF_TOPIC_SPLIT_SCHEMA_VERSION,
            "method": "explicit_train_val_test",
            "seed": _configured_seed(config),
            "n_splits": 1,
            "split_column": str(split_column),
            "fit_values": ["train", "val"],
            "heldout_value": "test",
            "fallback_reason": None,
        }
    raise ValueError(
        "multi_model_forest Stage 1 requires cv_folds > 1 or an explicit held-out test split"
    )


def _inner_split_plan(
    outer_train: pd.DataFrame,
    config: AppliedInferenceConfig,
    *,
    outer_fold: int,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Dict[str, Any]]:
    nn_config: MultiModelForestConfig = config.architecture.multi_model_forest
    inner_count = int(nn_config.candidate_consistency_inner_folds)
    inner_seed = _configured_seed(config) + 51_000 + int(outer_fold)
    return make_joint_treatment_outcome_splits(
        outer_train,
        treatment_column=config.treatment_column,
        outcome_column=config.outcome_column,
        outcome_type=config.outcome_type,
        n_splits=inner_count,
        seed=inner_seed,
    )


def build_tfidf_topic_split_provenance(
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
) -> List[Dict[str, Any]]:
    """Build the canonical Stage 1 outer/inner split provenance without fitting.

    Independent non-TF-IDF architecture runs use this function so every lane
    sees exactly the same rows as the full Stage 1 workflow.
    """

    data = dataset.reset_index(drop=True).copy()
    data["_oci_row_id"] = np.arange(len(data), dtype=int)
    identity = tfidf_topic_stage1_identity(config, data)
    dataset_identity = identity["dataset"]
    outer_splits, outer_metadata = _outer_split_plan(data, config)
    rows: List[Dict[str, Any]] = []
    for outer_fold, (train_idx, test_idx) in enumerate(outer_splits, start=1):
        outer_train = data.iloc[np.asarray(train_idx, dtype=int)].copy()
        outer_test = data.iloc[np.asarray(test_idx, dtype=int)].copy()
        row: Dict[str, Any] = {
            "split_schema_version": identity["split_semantics"]["schema_version"],
            "outer_fold": int(outer_fold),
            "fit_row_ids": outer_train["_oci_row_id"].astype(int).tolist(),
            "heldout_row_ids": outer_test["_oci_row_id"].astype(int).tolist(),
            "fit_row_fingerprint": row_set_fingerprint(outer_train["_oci_row_id"]),
            "heldout_row_fingerprint": row_set_fingerprint(outer_test["_oci_row_id"]),
            "honest_outer_holdout": True,
            "split_method": outer_metadata["method"],
            "split_seed": outer_metadata.get("seed"),
            "split_metadata": outer_metadata,
            "dataset_content_fingerprint": dataset_identity["content_fingerprint"],
            "dataset_ordered_row_fingerprint": dataset_identity[
                "ordered_row_fingerprint"
            ],
            "split_semantics_hash": identity["split_semantics_hash"],
            "inner_splits": [],
        }
        inner_splits, inner_metadata = _inner_split_plan(
            outer_train,
            config,
            outer_fold=int(outer_fold),
        )
        for inner_fold, (fit_local, heldout_local) in enumerate(inner_splits, start=1):
            inner_fit = outer_train.iloc[np.asarray(fit_local, dtype=int)]
            inner_heldout = outer_train.iloc[np.asarray(heldout_local, dtype=int)]
            row["inner_splits"].append(
                {
                    "inner_fold": int(inner_fold),
                    "fit_row_ids": inner_fit["_oci_row_id"].astype(int).tolist(),
                    "heldout_row_ids": inner_heldout["_oci_row_id"].astype(int).tolist(),
                    "fit_row_fingerprint": row_set_fingerprint(
                        inner_fit["_oci_row_id"]
                    ),
                    "heldout_row_fingerprint": row_set_fingerprint(
                        inner_heldout["_oci_row_id"]
                    ),
                    "split_method": inner_metadata["method"],
                    "split_seed": inner_metadata.get("seed"),
                    "split_metadata": inner_metadata,
                }
            )
        rows.append(row)
    return rows


def tfidf_topic_dataset_fingerprints(
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
) -> Dict[str, Any]:
    """Fingerprint ordered model inputs without incorporating oracle columns."""
    data = dataset.reset_index(drop=True)
    columns = [
        config.text_column,
        config.treatment_column,
        config.outcome_column,
    ]
    if config.split_column in data.columns:
        columns.append(config.split_column)
    columns = list(dict.fromkeys(columns))
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f"Cannot fingerprint Stage 1 dataset; missing columns: {missing}")
    model_inputs = data.loc[:, columns]
    try:
        row_hashes = pd.util.hash_pandas_object(
            model_inputs,
            index=False,
            categorize=True,
        ).to_numpy(dtype=np.uint64)
    except TypeError:
        # Model inputs are expected to be scalar, but stringify unusual object
        # cells deterministically rather than allowing an unsafe cache hit.
        row_hashes = pd.util.hash_pandas_object(
            model_inputs.map(lambda value: json.dumps(value, sort_keys=True, default=str)),
            index=False,
            categorize=True,
        ).to_numpy(dtype=np.uint64)
    schema = {
        "version": TFIDF_TOPIC_DATASET_FINGERPRINT_VERSION,
        "columns": columns,
        "dtypes": [str(model_inputs[column].dtype) for column in columns],
        "row_count": int(len(model_inputs)),
    }
    row_hash_values = [int(value) for value in row_hashes]
    return {
        **schema,
        "content_fingerprint": stable_hash({**schema, "row_hashes": sorted(row_hash_values)}),
        "ordered_row_fingerprint": stable_hash({**schema, "row_hashes": row_hash_values}),
    }


def tfidf_topic_split_semantics(config: AppliedInferenceConfig) -> Dict[str, Any]:
    """Return the versioned policy whose changes invalidate Stage 1 caches."""
    nn_config: MultiModelForestConfig = config.architecture.multi_model_forest
    return {
        "schema_version": TFIDF_TOPIC_SPLIT_SCHEMA_VERSION,
        "seed": _configured_seed(config),
        "outer": {
            "requested_folds": int(config.cv_folds),
            "preferred_method": "joint_treatment_outcome_stratified",
            "fallback_method": "shuffled_kfold_only_when_joint_strata_infeasible",
            "treatment_column": config.treatment_column,
            "outcome_column": config.outcome_column,
            "outcome_type": config.outcome_type,
            "fixed_split_column": config.split_column,
        },
        "inner": {
            "requested_folds": int(nn_config.candidate_consistency_inner_folds),
            "preferred_method": "joint_treatment_outcome_stratified",
            "fallback_method": "shuffled_kfold_only_when_joint_strata_infeasible",
            "seed_derivation": "config.seed + 51000 + outer_fold",
        },
    }


def tfidf_topic_stage1_identity(
    config: AppliedInferenceConfig,
    dataset: pd.DataFrame,
) -> Dict[str, Any]:
    """Describe every deterministic Stage 1 input used for cache identity."""
    nn_config: MultiModelForestConfig = config.architecture.multi_model_forest
    dataset_identity = tfidf_topic_dataset_fingerprints(dataset, config)
    split_semantics = tfidf_topic_split_semantics(config)
    orphan_enabled = bool(nn_config.tfidf_topic.orphan_ngram_enabled)
    return {
        "schema": HANDOFF_SCHEMA_VERSION,
        "discovery_schema": DISCOVERY_SCHEMA_VERSION,
        "orphan_ngram_score_schema": (
            ORPHAN_NGRAM_SCORE_SCHEMA_VERSION if orphan_enabled else None
        ),
        "orphan_nested_calibration_folds": (
            int(nn_config.tfidf_nested_calibration_folds) if orphan_enabled else None
        ),
        "views": [asdict(view) for view in nn_config.bow_views],
        "nuisance_folds": nn_config.nuisance_folds,
        "topic": asdict(nn_config.tfidf_topic),
        "text_column": config.text_column,
        "treatment_column": config.treatment_column,
        "outcome_column": config.outcome_column,
        "outcome_type": config.outcome_type,
        "dataset": dataset_identity,
        "split_semantics": split_semantics,
        "split_semantics_hash": stable_hash(split_semantics),
    }


def tfidf_topic_stage1_config_hash(
    config: AppliedInferenceConfig,
    dataset: pd.DataFrame,
) -> str:
    """Hash deterministic settings, model-input content, and split semantics."""
    return stable_hash(tfidf_topic_stage1_identity(config, dataset))


def _nested_calibration_plan(
    fit_df: pd.DataFrame,
    *,
    config: AppliedInferenceConfig,
    outer_fold: int,
    inner_fold: Optional[int],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Split only registered fit rows for orphan n-gram cluster selection.

    The registered held-out frame is deliberately not an argument.  A caller
    therefore cannot accidentally use its labels while choosing the nested
    split, topic count, terms, or orphan clusters.
    """

    if len(fit_df) < 6:
        raise ValueError("nested TF-IDF calibration requires at least six registered fit rows")
    nn_config: MultiModelForestConfig = config.architecture.multi_model_forest
    requested = max(2, int(nn_config.tfidf_nested_calibration_folds))
    # Each calibration fold needs at least two rows for covariance estimates,
    # while the model-training side needs at least four for nuisance OOF fits.
    maximum = max(2, len(fit_df) // 2)
    fold_count = min(requested, maximum)
    if len(fit_df) - int(np.ceil(len(fit_df) / fold_count)) < 4:
        raise ValueError("registered TF-IDF fit partition is too small for nested model training")
    nested_seed = _configured_seed(config) + 71_000 + 1_009 * int(outer_fold) + int(inner_fold or 0)
    splits, split_metadata = make_joint_treatment_outcome_splits(
        fit_df,
        treatment_column=config.treatment_column,
        outcome_column=config.outcome_column,
        outcome_type=config.outcome_type,
        n_splits=fold_count,
        seed=nested_seed,
    )
    # The selected fold is a fixed function of public fold coordinates and not
    # of any registered-heldout value.
    selected_index = (int(outer_fold) + int(inner_fold or 0) - 1) % len(splits)
    model_positions, calibration_positions = splits[selected_index]
    model_fit = fit_df.iloc[np.asarray(model_positions, dtype=int)].copy()
    calibration = fit_df.iloc[np.asarray(calibration_positions, dtype=int)].copy()
    if len(model_fit) < 4 or len(calibration) < 2:
        raise ValueError("nested TF-IDF model/calibration partition is infeasible")
    model_ids = model_fit["_oci_row_id"].astype(int).tolist()
    calibration_ids = calibration["_oci_row_id"].astype(int).tolist()
    registered_ids = fit_df["_oci_row_id"].astype(int).tolist()
    if set(model_ids) & set(calibration_ids) or set(model_ids) | set(calibration_ids) != set(
        registered_ids
    ):
        raise RuntimeError("nested TF-IDF calibration did not partition registered fit rows")
    return (
        model_fit,
        calibration,
        {
            "schema_version": TFIDF_NESTED_CALIBRATION_SCHEMA_VERSION,
            "policy": "nested_fit_calibration",
            "seed": nested_seed,
            "fold_count": fold_count,
            "configured_fold_count": requested,
            "fold_parameter": "tfidf_nested_calibration_folds",
            "canonical_hierarchy_partition_count_used": False,
            "interaction_inner_folds_used": False,
            "selected_fold": selected_index + 1,
            "split_method": split_metadata,
            "model_fit_row_ids": model_ids,
            "calibration_row_ids": calibration_ids,
            "model_fit_row_fingerprint": row_set_fingerprint(model_ids),
            "calibration_row_fingerprint": row_set_fingerprint(calibration_ids),
            "registered_heldout_labels_accessed": False,
            "nested_calibration_labels_accessed": True,
            "selection_frozen_before_registered_heldout_transform": True,
        },
    )


def _fit_tfidf_topic_context_with_orphan_calibration(
    *,
    spec: Dict[str, Any],
    config: AppliedInferenceConfig,
    artifact_dir: Path,
) -> Dict[str, Any]:
    """Fit one context on a nested model partition and select orphan clusters.

    The native model is trained on the nested model partition and orphan
    n-gram clusters are scored on the nested calibration partition of the
    registered fit rows.  Only after selection are the registered held-out
    texts predicted; their treatment and outcome columns are never selected or
    passed downstream.
    """

    nn_config: MultiModelForestConfig = config.architecture.multi_model_forest
    topic_config = nn_config.tfidf_topic
    registered_fit = spec["fit_df"].copy()
    registered_heldout = spec["heldout_df"].loc[:, ["_oci_row_id", config.text_column]].copy()
    model_fit, calibration, nesting = _nested_calibration_plan(
        registered_fit,
        config=config,
        outer_fold=int(spec["outer_fold"]),
        inner_fold=spec.get("inner_fold"),
    )
    metadata, stacks = fit_tfidf_topic_context(
        fit_df=model_fit,
        heldout_df=calibration,
        text_column=config.text_column,
        treatment_column=config.treatment_column,
        outcome_column=config.outcome_column,
        outcome_type=config.outcome_type,
        views=nn_config.bow_views,
        nuisance_folds=int(nn_config.nuisance_folds),
        config=topic_config,
        artifact_dir=artifact_dir,
        scope_id=f"{spec['scope_id']}__nested_calibration",
        score_orphan_ngrams=True,
    )
    branch = metadata.get("effect_orphan_ngram_branch") or {}
    if branch.get("status") != "completed":
        raise RuntimeError("nested TF-IDF orphan n-gram selection did not complete")

    # Selection is frozen; the fitted stacks may now predict registered heldout text.
    heldout_texts = registered_heldout[config.text_column].fillna("").tolist()
    registered_fit_ids = registered_fit["_oci_row_id"].astype(int).tolist()
    model_ids = nesting["model_fit_row_ids"]
    nuisance_path = Path(str(metadata["artifacts"]["nuisance_predictions"]))
    nested_nuisance = pd.read_parquet(nuisance_path)
    if set(map(int, nested_nuisance["_oci_row_id"])) != set(registered_fit_ids):
        raise RuntimeError("nested TF-IDF nuisance rows do not cover registered fit rows")
    nested_nuisance["prediction_scope"] = "fit_oof"
    e_heldout, e_views = stacks["treatment"].predict(heldout_texts)
    m_heldout, m_views = stacks["outcome"].predict(heldout_texts)
    external_rows: List[Dict[str, Any]] = []
    for position, row_id in enumerate(registered_heldout["_oci_row_id"].astype(int).tolist()):
        row: Dict[str, Any] = {
            "_oci_row_id": row_id,
            "prediction_scope": "external_heldout",
            "treatment_stacked": float(e_heldout[position]),
            "outcome_stacked": float(m_heldout[position]),
            "fit_row_ids": list(model_ids),
        }
        for view in nn_config.bow_views:
            row[f"treatment_view__{view.name}"] = float(e_views[view.name][position])
            row[f"outcome_view__{view.name}"] = float(m_views[view.name][position])
        external_rows.append(row)
    fit_order = {row_id: index for index, row_id in enumerate(registered_fit_ids)}
    nested_nuisance["_fit_order"] = nested_nuisance["_oci_row_id"].map(fit_order)
    nested_nuisance = nested_nuisance.sort_values("_fit_order").drop(columns="_fit_order")
    pd.concat(
        [nested_nuisance, pd.DataFrame(external_rows)],
        ignore_index=True,
        sort=False,
    ).to_parquet(nuisance_path, index=False)

    registered_heldout_ids = registered_heldout["_oci_row_id"].astype(int).tolist()
    metadata.update(
        {
            "scope_id": str(spec["scope_id"]),
            "fit_row_ids": registered_fit_ids,
            "heldout_row_ids": registered_heldout_ids,
            "fit_row_fingerprint": row_set_fingerprint(registered_fit_ids),
            "heldout_row_fingerprint": row_set_fingerprint(registered_heldout_ids),
            "model_fit_row_ids": list(model_ids),
            "model_fit_row_fingerprint": nesting["model_fit_row_fingerprint"],
            "orphan_ngram_label_policy": "nested_fit_calibration",
            "selection_nesting": nesting,
            "registered_heldout_columns_read": ["_oci_row_id", config.text_column],
            "registered_heldout_labels_accessed": False,
        }
    )
    return metadata


def _fit_tfidf_topic_stage1_spec_impl(
    spec: Dict[str, Any],
    *,
    contexts_dir: Path,
    config: AppliedInferenceConfig,
    stage1_hash: str,
    dataset_identity: Dict[str, Any],
    split_semantics_hash: str,
    split_schema_version: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Fit or reuse one exact context without capturing runner-local state."""
    nn_config: MultiModelForestConfig = config.architecture.multi_model_forest
    topic_config = nn_config.tfidf_topic
    orphan_enabled = bool(topic_config.orphan_ngram_enabled)
    metadata_path = contexts_dir / spec["scope_id"] / "context_metadata.json"
    if metadata_path.exists():
        try:
            cached = json.loads(metadata_path.read_text(encoding="utf-8"))
            artifacts = cached.get("artifacts") or {}
            artifact_paths = [
                artifacts.get("nuisance_predictions"),
                *((artifacts.get("ngram_scores") or {}).values()),
            ]
            if orphan_enabled:
                artifact_paths.append(artifacts.get("orphan_ngram_scores"))
            if (
                cached.get("fit_row_fingerprint")
                == row_set_fingerprint(spec["fit_df"]["_oci_row_id"])
                and cached.get("heldout_row_fingerprint")
                == row_set_fingerprint(spec["heldout_df"]["_oci_row_id"])
                and cached.get("config_hash") == stable_hash(asdict(topic_config))
                and cached.get("stage1_config_hash") == stage1_hash
                and cached.get("dataset_content_fingerprint")
                == dataset_identity["content_fingerprint"]
                and cached.get("dataset_ordered_row_fingerprint")
                == dataset_identity["ordered_row_fingerprint"]
                and cached.get("split_semantics_hash") == split_semantics_hash
                and all(path and Path(path).exists() for path in artifact_paths)
            ):
                nuisance = pd.read_parquet(artifacts["nuisance_predictions"])
                fit_rows = nuisance[
                    nuisance["prediction_scope"] == "fit_oof"
                ]["_oci_row_id"].astype(int)
                heldout_rows = nuisance[
                    nuisance["prediction_scope"] == "external_heldout"
                ]["_oci_row_id"].astype(int)
                if (
                    set(fit_rows) != set(map(int, cached["fit_row_ids"]))
                    or set(heldout_rows) != set(map(int, cached["heldout_row_ids"]))
                ):
                    raise ValueError("TF-IDF checkpoint nuisance row registry changed")
                logger.info(
                    "Reusing complete exact Stage 1 context scope_id=%s",
                    spec["scope_id"],
                )
                return spec, cached
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "Existing exact-context TF-IDF checkpoint failed validation "
                f"scope_id={spec['scope_id']}"
            ) from exc
    logger.info(
        "Stage 1 fitting exact context scope_id=%s fit=%s heldout=%s",
        spec["scope_id"],
        len(spec["fit_df"]),
        len(spec["heldout_df"]),
    )
    context_dir = contexts_dir / spec["scope_id"]
    if orphan_enabled:
        metadata = _fit_tfidf_topic_context_with_orphan_calibration(
            spec=spec,
            config=config,
            artifact_dir=context_dir,
        )
    else:
        metadata, _stacks = fit_tfidf_topic_context(
            fit_df=spec["fit_df"],
            heldout_df=spec["heldout_df"],
            text_column=config.text_column,
            treatment_column=config.treatment_column,
            outcome_column=config.outcome_column,
            outcome_type=config.outcome_type,
            views=nn_config.bow_views,
            nuisance_folds=int(nn_config.nuisance_folds),
            config=topic_config,
            artifact_dir=context_dir,
            scope_id=spec["scope_id"],
        )
    metadata.update(
        {
            "stage1_config_hash": stage1_hash,
            "dataset_content_fingerprint": dataset_identity["content_fingerprint"],
            "dataset_ordered_row_fingerprint": dataset_identity["ordered_row_fingerprint"],
            "split_semantics_hash": split_semantics_hash,
            "split_schema_version": split_schema_version,
        }
    )
    _write_json(metadata_path, metadata)
    logger.info("Stage 1 completed exact context scope_id=%s", spec["scope_id"])
    return spec, metadata


def _fit_tfidf_topic_stage1_spec(
    spec: Dict[str, Any],
    *,
    contexts_dir: Path,
    config: AppliedInferenceConfig,
    stage1_hash: str,
    dataset_identity: Dict[str, Any],
    split_semantics_hash: str,
    split_schema_version: str,
    limit_native_threads: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Pickle-safe joblib worker for one fold-isolated Stage 1 context."""
    from .discovery_randomness import (
        enable_deterministic_torch,
        seed_discovery_rngs,
    )

    expected_seed = _tfidf_context_scope_seed(
        global_seed=_configured_seed(config),
        scope_id=str(spec["scope_id"]),
    )
    if int(spec.get("worker_scope_seed", -1)) != expected_seed:
        raise ValueError("TF-IDF context worker seed binding changed")
    enable_deterministic_torch()
    seed_discovery_rngs(expected_seed, gpu_id=None)
    thread_limit = threadpool_limits(limits=1) if limit_native_threads else nullcontext()
    with thread_limit:
        return _fit_tfidf_topic_stage1_spec_impl(
            spec,
            contexts_dir=Path(contexts_dir),
            config=config,
            stage1_hash=stage1_hash,
            dataset_identity=dataset_identity,
            split_semantics_hash=split_semantics_hash,
            split_schema_version=split_schema_version,
        )


def _build_tfidf_worker_context_spec(
    *,
    outer_fold: int,
    inner_fold: int | None,
    scope: str,
    fold_key: int,
    fit_df: pd.DataFrame,
    heldout_df: pd.DataFrame,
    scope_id: str,
    config: AppliedInferenceConfig,
) -> Dict[str, Any]:
    """Build one serialized worker input; held-out rows carry only ID and text."""

    required = {"_oci_row_id", config.text_column}
    missing = sorted(required - set(heldout_df.columns))
    if missing:
        raise ValueError("TF-IDF heldout projection lacks ID/text: " + ", ".join(missing))
    serialized_heldout = heldout_df.loc[:, ["_oci_row_id", config.text_column]].copy()
    return {
        "outer_fold": int(outer_fold),
        "inner_fold": inner_fold,
        "scope": str(scope),
        "fold_key": int(fold_key),
        "fit_df": fit_df.copy(),
        "heldout_df": serialized_heldout,
        "scope_id": str(scope_id),
        "worker_scope_seed": _tfidf_context_scope_seed(
            global_seed=_configured_seed(config),
            scope_id=str(scope_id),
        ),
    }


def run_tfidf_topic_stage1(
    *,
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    output_path: Path,
    artifact_dir: Path,
    handoff_path: Path,
) -> None:
    """Fit every exact outer/inner context and write a v2 fail-closed handoff."""
    data = dataset.reset_index(drop=True).copy()
    data["_oci_row_id"] = np.arange(len(data), dtype=int)
    nn_config: MultiModelForestConfig = config.architecture.multi_model_forest
    contexts_dir = Path(artifact_dir) / "stage1_tfidf_topics" / "contexts"
    contexts_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    split_rows: List[Dict[str, Any]] = []
    primary_predictions: List[pd.DataFrame] = []
    all_context_specs: List[Dict[str, Any]] = []
    stage1_identity = tfidf_topic_stage1_identity(config, data)
    stage1_hash = stable_hash(stage1_identity)
    dataset_identity = stage1_identity["dataset"]
    split_semantics_hash = stage1_identity["split_semantics_hash"]
    outer_splits, outer_split_metadata = _outer_split_plan(data, config)

    for outer_fold, (train_idx, test_idx) in enumerate(outer_splits, start=1):
        train_idx = np.asarray(train_idx, dtype=int)
        test_idx = np.asarray(test_idx, dtype=int)
        outer_train = data.iloc[train_idx].copy()
        outer_test = data.iloc[test_idx].copy()
        outer_split_row = {
            "split_schema_version": stage1_identity["split_semantics"]["schema_version"],
            "outer_fold": int(outer_fold),
            "fit_row_ids": outer_train["_oci_row_id"].astype(int).tolist(),
            "heldout_row_ids": outer_test["_oci_row_id"].astype(int).tolist(),
            "fit_row_fingerprint": row_set_fingerprint(outer_train["_oci_row_id"]),
            "heldout_row_fingerprint": row_set_fingerprint(outer_test["_oci_row_id"]),
            "honest_outer_holdout": True,
            "split_method": outer_split_metadata["method"],
            "split_seed": outer_split_metadata.get("seed"),
            "split_metadata": outer_split_metadata,
            "dataset_content_fingerprint": dataset_identity["content_fingerprint"],
            "dataset_ordered_row_fingerprint": dataset_identity["ordered_row_fingerprint"],
            "split_semantics_hash": split_semantics_hash,
            "inner_splits": [],
        }

        inner_count = int(nn_config.candidate_consistency_inner_folds)
        if inner_count < 2 or len(outer_train) < inner_count:
            raise ValueError(
                "Exact candidate-selection CV requires at least two inner folds "
                "and at least one outer-training row per requested fold; "
                f"requested={inner_count} rows={len(outer_train)}"
            )
        inner_splits, inner_split_metadata = _inner_split_plan(
            outer_train,
            config,
            outer_fold=int(outer_fold),
        )
        context_specs: List[Dict[str, Any]] = []
        for inner_fold, (fit_local, heldout_local) in enumerate(inner_splits, start=1):
            fit_local = np.asarray(fit_local, dtype=int)
            heldout_local = np.asarray(heldout_local, dtype=int)
            inner_fit = outer_train.iloc[fit_local].copy()
            inner_heldout = outer_train.iloc[heldout_local].copy()
            outer_split_row["inner_splits"].append(
                {
                    "inner_fold": int(inner_fold),
                    "fit_row_ids": inner_fit["_oci_row_id"].astype(int).tolist(),
                    "heldout_row_ids": inner_heldout["_oci_row_id"].astype(int).tolist(),
                    "fit_row_fingerprint": row_set_fingerprint(inner_fit["_oci_row_id"]),
                    "heldout_row_fingerprint": row_set_fingerprint(inner_heldout["_oci_row_id"]),
                    "split_method": inner_split_metadata["method"],
                    "split_seed": inner_split_metadata.get("seed"),
                    "split_metadata": inner_split_metadata,
                }
            )
            context_specs.append(
                _build_tfidf_worker_context_spec(
                    outer_fold=int(outer_fold),
                    inner_fold=int(inner_fold),
                    scope="candidate_selection_inner_fit",
                    fold_key=1000 * int(outer_fold) + int(inner_fold),
                    fit_df=inner_fit,
                    heldout_df=inner_heldout,
                    scope_id=f"outer_{outer_fold:03d}_inner_{inner_fold:03d}",
                    config=config,
                )
            )
        context_specs.append(
            _build_tfidf_worker_context_spec(
                outer_fold=int(outer_fold),
                inner_fold=None,
                scope="full_outer_train",
                fold_key=int(outer_fold),
                fit_df=outer_train,
                heldout_df=outer_test,
                scope_id=f"outer_{outer_fold:03d}_full_train",
                config=config,
            )
        )
        split_rows.append(outer_split_row)
        all_context_specs.extend(context_specs)

    # Exact contexts share no fitted state. Schedule them globally instead of
    # serializing the outer folds, while limiting every worker to one numeric
    # thread. This preserves fold isolation and makes --cpus-total an actual
    # upper bound on concurrent context fits. Start the larger full-outer fits
    # first so the final worker wave contains only smaller inner contexts.
    all_context_specs.sort(
        key=lambda spec: (
            spec["scope"] != "full_outer_train",
            int(spec["outer_fold"]),
            int(spec.get("inner_fold") or 0),
        )
    )
    requested_workers = int(nn_config.cpus_total or 1)
    context_workers = max(1, min(len(all_context_specs), requested_workers))
    configured_backend, joblib_backend = _resolve_tfidf_topic_stage1_parallel_backend(
        getattr(nn_config, "outer_parallel_backend", "processes")
    )
    logger.info(
        "Stage 1 scheduling exact contexts=%s workers=%s backend=%s",
        len(all_context_specs),
        context_workers,
        configured_backend,
    )
    parallel_kwargs: Dict[str, Any] = {
        "backend": joblib_backend,
        "n_jobs": context_workers,
    }
    if joblib_backend == "loky":
        parallel_kwargs["inner_max_num_threads"] = 1
    with parallel_config(**parallel_kwargs):
        completed_contexts = Parallel(
            batch_size=1,
            pre_dispatch="all",
        )(
            delayed(_fit_tfidf_topic_stage1_spec)(
                spec,
                contexts_dir=contexts_dir,
                config=config,
                stage1_hash=stage1_hash,
                dataset_identity=dataset_identity,
                split_semantics_hash=split_semantics_hash,
                split_schema_version=stage1_identity["split_semantics"]["schema_version"],
                limit_native_threads=(joblib_backend in {"loky", "multiprocessing"}),
            )
            for spec in all_context_specs
        )

    full_metadata_by_outer: Dict[int, Dict[str, Any]] = {}
    for spec, metadata in completed_contexts:
        outer_fold = int(spec["outer_fold"])
        rows.append(
            {
                "schema_version": HANDOFF_SCHEMA_VERSION,
                "stage1_config_hash": stage1_hash,
                "fold_key": int(spec["fold_key"]),
                "outer_fold": outer_fold,
                "inner_fold": spec["inner_fold"],
                "scope": spec["scope"],
                "fit_row_ids": metadata["fit_row_ids"],
                "heldout_row_ids": metadata["heldout_row_ids"],
                "fit_row_fingerprint": metadata["fit_row_fingerprint"],
                "heldout_row_fingerprint": metadata["heldout_row_fingerprint"],
                "dataset_content_fingerprint": dataset_identity["content_fingerprint"],
                "dataset_ordered_row_fingerprint": dataset_identity["ordered_row_fingerprint"],
                "split_semantics_hash": split_semantics_hash,
                "discovery": metadata,
            }
        )
        if spec["scope"] == "full_outer_train":
            if outer_fold in full_metadata_by_outer:
                raise RuntimeError(f"Duplicate full outer context completed for fold {outer_fold}")
            full_metadata_by_outer[outer_fold] = metadata

    for outer_fold in sorted({int(row["outer_fold"]) for row in split_rows}):
        full_metadata = full_metadata_by_outer.get(outer_fold)
        if full_metadata is None:
            raise RuntimeError(f"Full outer context did not complete for fold {outer_fold}")
        nuisance = pd.read_parquet(full_metadata["artifacts"]["nuisance_predictions"])
        nuisance = nuisance[nuisance["prediction_scope"] == "external_heldout"].copy()
        nuisance["outer_fold"] = int(outer_fold)
        nuisance["honest_outer_holdout"] = True
        nuisance["estimation_provenance"] = "outer_train_tfidf_nuisance_only"
        primary_predictions.append(nuisance)

    rows.sort(key=lambda row: (int(row["outer_fold"]), int(row.get("inner_fold") or 999)))
    required_inner = int(nn_config.candidate_consistency_inner_folds)
    for outer_fold in sorted({int(row["outer_fold"]) for row in rows}):
        inner_rows = [
            row
            for row in rows
            if int(row["outer_fold"]) == outer_fold
            and row["scope"] == "candidate_selection_inner_fit"
        ]
        full_rows = [
            row
            for row in rows
            if int(row["outer_fold"]) == outer_fold and row["scope"] == "full_outer_train"
        ]
        if len(inner_rows) != required_inner or len(full_rows) != 1:
            raise RuntimeError(
                f"Exact Stage 1 context set is incomplete for outer_fold={outer_fold}: "
                f"inner={len(inner_rows)}/{required_inner}, full={len(full_rows)}/1"
            )

    handoff_path = Path(handoff_path)
    _write_jsonl(handoff_path, rows)
    _write_json(
        handoff_path.parent / "manifest.json",
        {
            "schema_version": HANDOFF_SCHEMA_VERSION,
            "stage1_config_hash": stage1_hash,
            "dataset_content_fingerprint": dataset_identity["content_fingerprint"],
            "dataset_ordered_row_fingerprint": dataset_identity["ordered_row_fingerprint"],
            "split_semantics_hash": split_semantics_hash,
            "split_schema_version": stage1_identity["split_semantics"]["schema_version"],
            "path": str(handoff_path),
            "n_rows": len(rows),
            "n_outer_folds": len(split_rows),
            "inner_contexts_per_outer": required_inner,
            "exact_inner_contexts": True,
            "parallel_context_workers": context_workers,
            "parallel_context_backend": configured_backend,
            "stage1_raw_text_forest_prediction": False,
            "stage2_raw_text_modeling_required": False,
            "orphan_ngram_clusters": bool(nn_config.tfidf_topic.orphan_ngram_enabled),
            "outer_test_labels_used": False,
            "feature_discovery_methods": ["bow", "tfidf_topic_contrast"],
        },
    )
    _write_jsonl(Path(artifact_dir) / "split_provenance.jsonl", split_rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(primary_predictions, ignore_index=True).sort_values("_oci_row_id").to_parquet(
        output_path, index=False
    )
    logger.info(
        "Saved Stage 1 nuisance/topic handoff contexts=%s path=%s",
        len(rows),
        handoff_path,
    )
