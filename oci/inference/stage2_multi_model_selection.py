"""Resampled, fold-honest modeling evidence; numerical results never gate roles.

Every candidate is exposed to every family. Forest feature subsets partition a
fresh shuffled candidate list, so absence from a model is not a negative vote.
Nuisances are elastic nets, cross-fitted inside each inner training partition;
they stay fixed across its stability perturbations. Outer-test data and oracle
columns are never inspected. Only aggregate summaries enter adjudication.
"""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import logging
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from threadpoolctl import threadpool_limits

from . import stage2_elastic_net_selection as linear
from .stage2_multi_model_config import FAMILIES, SCHEMA_VERSION
from .stage2_role_adjudication import _fingerprint, _write_json
from .stage2_statistical_selection import _modifier_test_chunk

LOGGER = logging.getLogger(__name__)
ROLES = ("treatment", "outcome", "effect")


class NotEstimable(ValueError):
    """An expected statistical limitation, recorded separately from a zero."""


def _key(feature: Mapping[str, Any]) -> str:
    return str(feature.get("feature_id") or feature["name"])


def _frame_hash(frame: pd.DataFrame) -> str:
    # Include Python scalar types: True, 1, and '1' are different categories.
    values = [
        [(type(x).__name__, repr(x)) for x in row]
        for row in frame.itertuples(index=False, name=None)
    ]
    return _fingerprint({"columns": list(frame.columns), "values": values})


def numerical_identity(
    *, frame, labels, definitions, inner_splits, outcome_type, seed, policy
):
    """Count-search settings do not change the underlying numerical evidence."""
    numerical_policy = policy.public_dict()
    numerical_policy["multi_model"].pop("modifier_count", None)
    return {
        "schema_version": SCHEMA_VERSION,
        "policy": numerical_policy,
        "seed": int(seed),
        "definitions": definitions,
        "inner_splits": list(inner_splits),
        "measurements_sha256": _frame_hash(frame),
        "observed_labels_sha256": _frame_hash(labels),
        "outcome_type": outcome_type,
        "component_source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "linear_component_source_sha256": sha256(Path(linear.__file__).read_bytes()).hexdigest(),
    }


def _checkpoint(
    directory: Path | None, name: str, fingerprint: str, compute: Callable[[], dict[str, Any]]
) -> dict[str, Any]:
    path = directory / f"{name}.json" if directory is not None else None
    if path is not None and path.is_file():
        cached = json.loads(path.read_text())
        if cached.get("input_fingerprint") != fingerprint:
            raise ValueError(f"incompatible multi-model checkpoint: {path}")
        if cached.get("result_sha256") != _fingerprint(cached.get("result")):
            raise ValueError(f"corrupt multi-model checkpoint: {path}")
        return cached["result"]
    result = compute()
    if path is not None:
        _write_json(
            path,
            {
                "input_fingerprint": fingerprint,
                "result_sha256": _fingerprint(result),
                "result": result,
            },
        )
    return result


def _validated_inputs(dataset, extracted, definitions, splits, treatment_column, outcome_column):
    names = [str(f["name"]) for f in definitions]
    ids = [_key(f) for f in definitions]
    if len(set(ids)) != len(ids) or len(set(names)) != len(names):
        raise ValueError("multi_model requires unique feature IDs and measurement names")
    if "_oci_row_id" in names or not set(names).issubset(extracted.columns):
        raise ValueError("multi_model requires every candidate measurement column")
    row_ids = extracted["_oci_row_id"].tolist()
    if any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) for x in row_ids):
        raise ValueError("multi_model row IDs must be integers")
    pool = set(row_ids)
    if len(pool) != len(row_ids) or not pool or min(pool) < 0 or max(pool) >= len(dataset):
        raise ValueError("multi_model requires unique valid outer-training row IDs")
    validation_counts = Counter()
    fold_ids = []
    for i, split in enumerate(splits, 1):
        train, valid = list(split.get("fit_row_ids", [])), list(split.get("heldout_row_ids", []))
        if (
            not train
            or not valid
            or len(set(train)) != len(train)
            or len(set(valid)) != len(valid)
            or set(train) & set(valid)
            or set(train) | set(valid) != pool
        ):
            raise ValueError("inner folds must partition exactly the supplied outer-training rows")
        validation_counts.update(valid)
        fold_ids.append(int(split.get("inner_fold", i)))
    if (
        len(set(fold_ids)) != len(fold_ids)
        or set(validation_counts) != pool
        or set(validation_counts.values()) != {1}
    ):
        raise ValueError("inner validation folds must cover each training patient exactly once")
    # This is the only access to the source dataset. No text, oracle, or other columns.
    labels = dataset.iloc[row_ids][[treatment_column, outcome_column]].copy()
    labels.index = row_ids
    if not np.isfinite(labels.to_numpy(dtype=float)).all():
        raise ValueError("treatment and outcome must be finite")
    if set(labels[treatment_column].unique()) != {0, 1}:
        raise ValueError("multi_model requires both binary treatment arms")
    return extracted[["_oci_row_id", *names]].set_index("_oci_row_id", drop=False), labels


def _matrix(design, treatment, kind, *, valid=False):
    x = design.valid if valid else design.train
    keys = list(design.column_feature_ids)
    if kind == "main":
        return x, keys, ["main"] * len(keys), keys
    t = np.asarray(treatment, dtype=float).reshape(-1, 1)
    if kind == "interactions":
        return (
            np.column_stack((t, x, t * x)),
            ["__treatment__", *keys, *keys],
            ["treatment", *(["main"] * len(keys)), *(["effect"] * len(keys))],
            ["treatment", *(f"main:{k}" for k in keys), *(f"effect:{k}" for k in keys)],
        )
    if kind == "rlearner":
        return (
            np.column_stack((t, t * x)),
            ["__treatment__", *keys],
            ["treatment", *(["effect"] * len(keys))],
            ["treatment", *(f"effect:{k}" for k in keys)],
        )
    raise ValueError(kind)


def _fit_linear(
    train, valid, definitions, y, *, binary, kind, treatment, valid_treatment, policy, seed, ratio
):
    """Tune inside training, refitting encoders in every penalty-CV partition."""
    y = np.asarray(y, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    alphas = np.logspace(
        policy.minimum_log10_alpha,
        policy.maximum_log10_alpha,
        policy.multi_model.regularization_grid_size,
    )[::-1]
    splits = linear._crossfit_indices(
        y if binary else treatment,
        requested_folds=policy.internal_cv_folds,
        seed=seed,
    )
    final_design = linear._encode_design(
        train, valid, definitions, categorical_min_count=policy.categorical_min_count
    )
    x, keys, blocks, groups = _matrix(final_design, treatment, kind)
    xv, _, _, _ = _matrix(final_design, valid_treatment, kind, valid=True)
    intercept = kind != "rlearner"
    if (binary and len(np.unique(y)) < 2) or x.shape[1] == 0:
        constant = float(np.clip(np.mean(y), 1e-6, 1 - 1e-6) if binary else np.mean(y))
        return {
            "prediction": np.full(len(valid), constant),
            "coefficients": np.zeros(x.shape[1]),
            "design": final_design,
            "keys": keys,
            "blocks": blocks,
            "state": None,
            "audit": {
                "status": "constant_target_or_empty_design",
                "converged": True,
                "iterations": 0,
                "regularization": None,
                "l1_ratio": ratio,
                "cv_folds": 0,
            },
        }
    if not splits:
        raise NotEstimable("insufficient_rows_for_penalty_cv")
    losses = np.full((len(splits), len(alphas)), np.inf)
    cv_iterations = []
    for fit, hold in splits:
        design = linear._encode_design(
            train.iloc[fit],
            train.iloc[hold],
            definitions,
            categorical_min_count=policy.categorical_min_count,
        )
        fit_x, _, _, group_ids = _matrix(design, treatment[fit], kind)
        hold_x, _, _, _ = _matrix(design, treatment[hold], kind, valid=True)
        group_structure = linear._group_structure(group_ids)
        initial = None
        fold_index = len(cv_iterations)
        iterations = []
        for a, alpha in enumerate(alphas):
            state = linear._fit_group_elastic_net(
                fit_x,
                y[fit],
                groups=group_structure,
                regularization=float(alpha),
                group_ratio=ratio,
                binary=binary,
                fit_intercept=intercept,
                max_iter=policy.max_iter,
                tolerance=policy.optimization_tolerance,
                initial=initial,
            )
            initial = state
            iterations.append({"iterations": state.iterations, "converged": state.converged})
            if state.converged:
                prediction = linear._state_prediction(state, hold_x, binary=binary)
                losses[fold_index, a] = linear._loss(y[hold], prediction, binary=binary)
        cv_iterations.append(iterations)
    means = losses.mean(axis=0)
    if not np.isfinite(means).any():
        raise NotEstimable("no_regularization_value_converged_in_every_cv_fold")
    best = int(np.argmin(means))  # Descending alpha breaks ties toward stronger regularization.
    state = linear._fit_group_elastic_net(
        x,
        y,
        groups=linear._group_structure(groups),
        regularization=float(alphas[best]),
        group_ratio=ratio,
        binary=binary,
        fit_intercept=intercept,
        max_iter=policy.max_iter,
        tolerance=policy.optimization_tolerance,
    )
    if not state.converged:
        raise NotEstimable("final_penalized_fit_did_not_converge")
    return {
        "prediction": linear._state_prediction(state, xv, binary=binary),
        "coefficients": state.coefficients,
        "design": final_design,
        "keys": keys,
        "blocks": blocks,
        "state": state,
        "audit": {
            "status": "ok",
            "converged": True,
            "iterations": state.iterations,
            "regularization": float(alphas[best]),
            "l1_ratio": ratio,
            "cv_folds": len(splits),
            "cv_loss": float(means[best]),
            "selected_on_grid_boundary": best in {0, len(alphas) - 1},
            "cv_optimizer_audit": cv_iterations,
        },
    }


def _nuisances(train, valid, definitions, t, y, *, binary, policy, seed):
    """All-candidate elastic nets with nested, training-only imputation and tuning."""
    e = np.full(len(t), np.nan)
    m = np.full(len(y), np.nan)
    audits = []
    subfolds = linear._crossfit_indices(t, requested_folds=policy.internal_cv_folds, seed=seed)
    if not subfolds:
        raise NotEstimable("insufficient_rows_for_nuisance_cross_fitting")
    for j, (fit, hold) in enumerate(subfolds):
        pair = []
        for role, target, is_binary in (("treatment", t, True), ("outcome", y, binary)):
            model = _fit_linear(
                train.iloc[fit],
                train.iloc[hold],
                definitions,
                target[fit],
                binary=is_binary,
                kind="main",
                treatment=t[fit],
                valid_treatment=t[hold],
                policy=policy,
                seed=seed + j * 10,
                ratio=policy.l1_ratio,
            )
            (e if role == "treatment" else m)[hold] = model["prediction"]
            pair.append({"role": role, **model["audit"]})
        audits.append(
            {
                "fit_row_ids": train.iloc[fit]._oci_row_id.astype(int).tolist(),
                "heldout_row_ids": train.iloc[hold]._oci_row_id.astype(int).tolist(),
                "models": pair,
            }
        )
    full = []
    for target, is_binary in ((t, True), (y, binary)):
        full.append(
            _fit_linear(
                train,
                valid,
                definitions,
                target,
                binary=is_binary,
                kind="main",
                treatment=t,
                valid_treatment=np.zeros(len(valid)),
                policy=policy,
                seed=seed + 99,
                ratio=policy.l1_ratio,
            )
        )
    if not np.isfinite(e).all() or not np.isfinite(m).all():
        raise RuntimeError("nuisance cross-fitting did not predict every training row")
    return {
        "training_propensity": e.tolist(),
        "training_outcome": m.tolist(),
        "validation_propensity": full[0]["prediction"].tolist(),
        "validation_outcome": full[1]["prediction"].tolist(),
        "crossfit_audit": audits,
        "validation_fit_audits": [x["audit"] for x in full],
    }


def _record(feature_id, role, *, status="ok", selected=None, score=None, **details):
    return {
        "feature_id": feature_id,
        "role": role,
        "status": status,
        "selected": selected,
        "score": score,
        **details,
    }


def _coefficient_records(model, definitions, *, block, role, policy):
    result = []
    for feature in definitions:
        indices = [
            i
            for i, (key, b) in enumerate(zip(model["keys"], model["blocks"]))
            if key == _key(feature) and b == block
        ]
        if model["state"] is None:
            indices = []  # A constant target supplies no coefficient-selection evidence.
        norm = float(np.linalg.norm(model["coefficients"][indices])) if indices else None
        result.append(
            _record(
                _key(feature),
                role,
                status="ok" if indices else "not_evaluable",
                selected=norm > policy.coefficient_tolerance if norm is not None else None,
                score=norm,
            )
        )
    return result


def _subsample(treatment, fraction, rng):
    indices = []
    for label in (0, 1):
        group = np.flatnonzero(treatment == label)
        if len(group):
            count = min(len(group), max(2, int(np.ceil(fraction * len(group)))))
            indices.extend(rng.choice(group, count, replace=False).tolist())
    return np.asarray(sorted(indices), dtype=int)


def candidate_subsets(feature_ids, *, repeat, subset_size, seed):
    """Full candidate pool once, then balanced random partitions with full coverage."""
    if repeat == 0:
        return [list(feature_ids)]
    shuffled = np.random.default_rng(seed).permutation(feature_ids).tolist()
    return [shuffled[i : i + subset_size] for i in range(0, len(shuffled), subset_size)]


def grouped_permutation_gain(x, column_ids, score, *, repeats, seed):
    """Permute all encoded columns of a candidate together on validation rows."""
    baseline = float(score(x))
    rng = np.random.default_rng(seed)
    result = {}
    for feature_id in dict.fromkeys(column_ids):
        columns = np.flatnonzero(np.asarray(column_ids) == feature_id)
        gains = []
        for _ in range(repeats):
            permuted = x.copy()
            permuted[:, columns] = x[rng.permutation(len(x))][:, columns]
            gains.append(float(score(permuted)) - baseline)
        result[feature_id] = {"gain": float(np.mean(gains)), "sd": float(np.std(gains))}
    return baseline, result


def _univariable(train, definitions, t, y, *, binary, policy):
    records = []
    intercept = np.ones((len(t), 1))
    adjusted, _ = linear._univariable_rank_safe_columns(intercept, t.reshape(-1, 1))

    def tested_p(call):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = call()
        unreliable = any(
            any(
                term in str(w.message).lower()
                for term in ("converg", "separation", "overflow", "divide by zero", "invalid value")
            )
            for w in caught
        )
        return None if unreliable else value

    for feature in definitions:
        # Rare levels and separation are unevaluable evidence, not zero support.
        design = linear._encode_univariable_feature(train, feature)
        outcome_test = (
            linear._binary_nested_p_value if binary else linear._continuous_nested_p_value
        )
        values = (
            tested_p(lambda: linear._binary_nested_p_value(t, intercept, design.main)[0]),
            tested_p(lambda: outcome_test(y, adjusted, design.main)[0]),
            tested_p(
                lambda: _modifier_test_chunk(
                    train,
                    t,
                    y,
                    np.column_stack((np.ones(len(t)), t)),
                    [feature],
                    binary_outcome=binary,
                    p_value_threshold=policy.multi_model.nominal_p_threshold,
                )[0]["interaction_p_value"]
            ),
        )
        for role, p in zip(ROLES, values):
            records.append(
                _record(
                    _key(feature),
                    role,
                    status="ok" if p is not None else "not_evaluable",
                    selected=p < policy.multi_model.nominal_p_threshold if p is not None else None,
                    score=-float(np.log10(max(p, 1e-300))) if p is not None else None,
                    p_value=p,
                )
            )
    for role in ROLES:
        rows = [r for r in records if r["role"] == role]
        for row, q in zip(rows, linear._benjamini_hochberg([r["p_value"] for r in rows])):
            row["q_value"] = q
            row["q_supported"] = q < policy.multi_model.q_threshold if q is not None else None
    return {
        "records": records,
        "audit": {
            "modifier_model": "Y ~ T + candidate + T:candidate",
            "outcome_association_model": "Y ~ T + candidate",
            "multiplicity": "BH_within_resample_and_endpoint",
            "hard_gate": False,
        },
    }


def _penalized_family(
    family, train, valid, definitions, t, y, tv, yv, *, binary, policy, seed, ratio, nuisance
):
    records, audits = [], []
    targets = (("treatment", t, tv, True), ("outcome", y, yv, binary))
    kind = "main"
    if family == "penalized_interactions":
        targets = (("outcome", y, yv, binary),)
        kind = "interactions"
    elif family == "orthogonal_linear":
        targets = (("effect", y - nuisance["m"], yv - nuisance["mv"], False),)
        t, tv = t - nuisance["e"], tv - nuisance["ev"]
        kind = "rlearner"
    for role, target, validation_target, is_binary in targets:
        try:
            model = _fit_linear(
                train,
                valid,
                definitions,
                target,
                binary=is_binary,
                kind=kind,
                treatment=t,
                valid_treatment=tv,
                policy=policy,
                seed=seed,
                ratio=ratio,
            )
            loss = linear._loss(validation_target, model["prediction"], binary=is_binary)
            if kind == "rlearner":
                constant = float(t @ target / max(float(t @ t), 1e-12))
                baseline_prediction = tv * constant
            else:
                baseline_prediction = np.full(len(valid), np.mean(target))
            model_gain = (
                linear._loss(validation_target, baseline_prediction, binary=is_binary) - loss
            )
            audits.append(
                {
                    "role": role,
                    "heldout_loss": loss,
                    "heldout_gain_over_constant": model_gain,
                    **model["audit"],
                }
            )
            rows = _coefficient_records(
                model,
                definitions,
                block="effect" if role == "effect" else "main",
                role=role,
                policy=policy,
            )
            records.extend({**r, "model_gain": model_gain} for r in rows)
            if kind == "interactions":
                # Outcome CV selected the penalty, but effect evidence also records
                # its independent held-out R-loss relative to a constant effect.
                effect_gain = None
                if model["state"] is not None:
                    x0, _, _, _ = _matrix(model["design"], np.zeros(len(valid)), kind, valid=True)
                    x1, _, _, _ = _matrix(model["design"], np.ones(len(valid)), kind, valid=True)
                    tau = linear._state_prediction(
                        model["state"], x1, binary=binary
                    ) - linear._state_prediction(model["state"], x0, binary=binary)
                    tr, yr = t - nuisance["e"], y - nuisance["m"]
                    tvr, yvr = tv - nuisance["ev"], yv - nuisance["mv"]
                    constant = float(tr @ yr / max(float(tr @ tr), 1e-12))
                    effect_gain = float(
                        np.mean((yvr - tvr * constant) ** 2) - np.mean((yvr - tvr * tau) ** 2)
                    )
                records.extend(
                    {**r, "model_gain": effect_gain}
                    for r in _coefficient_records(
                        model, definitions, block="effect", role="effect", policy=policy
                    )
                )
        except NotEstimable as exc:
            roles = (role, "effect") if kind == "interactions" else (role,)
            records.extend(
                _record(_key(f), r, status="not_evaluable") for f in definitions for r in roles
            )
            audits.append({"role": role, "status": str(exc)})
    return {"records": records, "audit": audits}


def _univariable_rlearner(train, valid, definitions, tr, yr, tvr, yvr, *, policy):
    design = linear._encode_design(
        train, valid, definitions, categorical_min_count=policy.categorical_min_count
    )
    denominator = float(tr @ tr)
    if denominator <= 1e-12:
        raise NotEstimable("no_residual_treatment_variation")
    constant = float(tr @ yr / denominator)
    baseline = float(np.mean((yvr - tvr * constant) ** 2))
    records = []
    for f in definitions:
        columns = np.flatnonzero(np.asarray(design.column_feature_ids) == _key(f))
        if not len(columns):
            records.append(_record(_key(f), "effect", status="not_evaluable"))
            continue
        x = np.column_stack((tr, tr[:, None] * design.train[:, columns]))
        xv = np.column_stack((tvr, tvr[:, None] * design.valid[:, columns]))
        penalty = np.diag([0.0, *([policy.modifier_ridge_alpha] * len(columns))])
        beta = np.linalg.lstsq(x.T @ x + penalty, x.T @ yr, rcond=None)[0]
        loss = float(np.mean((yvr - xv @ beta) ** 2))
        gain = baseline - loss
        records.append(_record(_key(f), "effect", selected=gain > 0, score=gain))
    return {
        "records": records,
        "audit": {"baseline_heldout_r_loss": baseline, "ridge_alpha": policy.modifier_ridge_alpha},
    }


def _forests(family, train, valid, definitions, t, y, tv, yv, *, binary, policy, seed, nuisance):
    design = linear._encode_design(
        train, valid, definitions, categorical_min_count=policy.categorical_min_count
    )
    if not design.train.shape[1]:
        raise NotEstimable("no_variable_candidate_columns")
    cfg = policy.multi_model
    records, audits = [], []
    tasks = ("effect",) if family == "causal_forest" else ("treatment", "outcome")
    for role in tasks:
        if role == "effect":
            from econml.grf import CausalForest

            tr, yr = t - nuisance["e"], y - nuisance["m"]
            tvr, yvr = tv - nuisance["ev"], yv - nuisance["mv"]
            if len(train) < 4 * cfg.forest_min_samples_leaf or float(tr @ tr) <= 1e-12:
                raise NotEstimable("insufficient_rows_or_residual_treatment_variation_for_forest")
            model = CausalForest(
                n_estimators=cfg.forest_trees,
                min_samples_leaf=cfg.forest_min_samples_leaf,
                max_samples=0.45,
                max_features=1.0,
                honest=True,
                inference=False,
                n_jobs=1,
                random_state=seed,
            )
            model.fit(design.train, tr.reshape(-1, 1), yr)

            def score(x):
                return np.mean((yvr - tvr * model.predict(x).reshape(-1)) ** 2)

        else:
            is_binary = role == "treatment" or binary
            target, target_valid = (t, tv) if role == "treatment" else (y, yv)
            if is_binary and len(np.unique(target)) < 2:
                records.extend(_record(_key(f), role, status="not_evaluable") for f in definitions)
                audits.append({"role": role, "status": "constant_target"})
                continue
            cls = RandomForestClassifier if is_binary else RandomForestRegressor
            model = cls(
                n_estimators=cfg.forest_trees,
                min_samples_leaf=cfg.forest_min_samples_leaf,
                max_features=0.7,
                n_jobs=1,
                random_state=seed,
            )
            model.fit(design.train, target)

            def score(x):
                prediction = model.predict_proba(x)[:, 1] if is_binary else model.predict(x)
                return linear._loss(target_valid, prediction, binary=is_binary)

        baseline, gains = grouped_permutation_gain(
            design.valid,
            design.column_feature_ids,
            score,
            repeats=cfg.permutation_repeats,
            seed=seed + 20,
        )
        importance = np.asarray(model.feature_importances_)
        if role == "effect":
            constant = float(tr @ yr / max(float(tr @ tr), 1e-12))
            model_gain = float(np.mean((yvr - tvr * constant) ** 2)) - baseline
        else:
            model_gain = (
                linear._loss(
                    target_valid, np.full(len(target_valid), np.mean(target)), binary=is_binary
                )
                - baseline
            )
        for f in definitions:
            feature_id = _key(f)
            columns = np.flatnonzero(np.asarray(design.column_feature_ids) == feature_id)
            gain = gains.get(feature_id)
            records.append(
                _record(
                    feature_id,
                    role,
                    status="ok" if gain else "not_evaluable",
                    selected=gain["gain"] > 0 if gain else None,
                    score=gain["gain"] if gain else None,
                    permutation_sd=gain["sd"] if gain else None,
                    model_gain=model_gain,
                    split_importance=float(importance[columns].sum()) if len(columns) else None,
                )
            )
        audits.append(
            {
                "role": role,
                "heldout_loss": baseline,
                "trees": cfg.forest_trees,
                "min_samples_leaf": cfg.forest_min_samples_leaf,
                "nuisance_model_family": (
                    "elastic_net" if role == "effect" else "not_a_nuisance_model"
                ),
            }
        )
    return {"records": records, "audit": audits}


def aggregate_evidence(cells, feature_ids):
    """Separate family denominators and fold counts; never count a missing fit as a failure vote."""
    result = {feature_id: [] for feature_id in feature_ids}
    for family in FAMILIES:
        by_key = {}
        for cell in cells:
            if cell["family"] != family:
                continue
            for record in cell["records"]:
                by_key.setdefault((record["feature_id"], record["role"]), []).append((cell, record))
        for (feature_id, role), values in by_key.items():
            evaluable = [(c, r) for c, r in values if r["status"] == "ok"]
            scores = [r["score"] for _, r in evaluable if r["score"] is not None]
            fold_counts = []
            for fold in sorted({c["inner_fold"] for c, _ in values}):
                observed = [r for c, r in evaluable if c["inner_fold"] == fold]
                fold_counts.append(
                    {
                        "inner_fold": fold,
                        "evaluated": len(observed),
                        "supported": sum(r["selected"] is True for r in observed),
                    }
                )
            p_values = [r["p_value"] for _, r in evaluable if r.get("p_value") is not None]
            q_values = [r["q_value"] for _, r in evaluable if r.get("q_value") is not None]
            result[feature_id].append(
                {
                    "evidence_id": f"multi:{feature_id}:{family}:{role}",
                    "family": family,
                    "role": role,
                    "exposures": len(values),
                    "evaluated": len(evaluable),
                    "not_evaluable": len(values) - len(evaluable),
                    "supported": sum(r["selected"] is True for _, r in evaluable),
                    "support_fraction": (
                        float(np.mean([r["selected"] for _, r in evaluable])) if evaluable else None
                    ),
                    "mean_score": float(np.mean(scores)) if scores else None,
                    "min_score": float(np.min(scores)) if scores else None,
                    "max_score": float(np.max(scores)) if scores else None,
                    "median_p": float(np.median(p_values)) if p_values else None,
                    "median_q": float(np.median(q_values)) if q_values else None,
                    "q_supported": sum(r.get("q_supported") is True for _, r in evaluable),
                    **{
                        f"mean_{key}": (
                            float(np.mean([r[key] for _, r in evaluable if r.get(key) is not None]))
                            if any(r.get(key) is not None for _, r in evaluable)
                            else None
                        )
                        for key in ("model_gain", "split_importance", "permutation_sd")
                    },
                    "folds": fold_counts,
                }
            )
    return result


def select_stage2_features_multi_model(
    *,
    dataset,
    extracted_fit,
    definitions,
    inner_splits,
    treatment_column,
    outcome_column,
    outcome_type,
    seed,
    policy,
    checkpoint_dir=None,
):
    policy.validate()
    if outcome_type not in {"binary", "continuous"}:
        raise ValueError("multi_model outcome_type must be binary or continuous")
    definitions = [dict(f) for f in definitions]
    if not definitions:
        return (
            [],
            {
                "schema_version": SCHEMA_VERSION,
                "policy": policy.public_dict(),
                "status": "complete_no_candidates",
                "decisions": [],
            },
            [],
            [],
        )
    frame, labels = _validated_inputs(
        dataset, extracted_fit, definitions, inner_splits, treatment_column, outcome_column
    )
    binary = outcome_type == "binary"
    if binary and not set(labels[outcome_column].unique()).issubset({0, 1}):
        raise ValueError("binary outcome must contain only 0 and 1")
    identity = numerical_identity(
        frame=frame,
        labels=labels,
        definitions=definitions,
        inner_splits=inner_splits,
        outcome_type=outcome_type,
        seed=seed,
        policy=policy,
    )
    fingerprint = _fingerprint(identity)
    directory = Path(checkpoint_dir) / fingerprint[:20] if checkpoint_dir is not None else None
    if directory is not None:
        _write_json(directory / "input.json", {**identity, "input_fingerprint": fingerprint})
    cells, oof = [], []
    cfg = policy.multi_model
    by_id = {_key(f): f for f in definitions}
    with threadpool_limits(limits=1):
        for position, split in enumerate(inner_splits, 1):
            fold = int(split.get("inner_fold", position))
            train_ids, valid_ids = split["fit_row_ids"], split["heldout_row_ids"]
            train, valid = frame.loc[train_ids].reset_index(drop=True), frame.loc[
                valid_ids
            ].reset_index(drop=True)
            t, y = labels.loc[train_ids, [treatment_column, outcome_column]].to_numpy(dtype=float).T
            tv, yv = (
                labels.loc[valid_ids, [treatment_column, outcome_column]].to_numpy(dtype=float).T
            )
            fold_seed = int(seed) + position * 10_000
            LOGGER.info("Stage 2 multi-model fold=%s: nested elastic-net nuisances", fold)
            nuisance = _checkpoint(
                directory,
                f"fold_{fold:03d}/nuisances",
                fingerprint,
                lambda: _nuisances(
                    train, valid, definitions, t, y, binary=binary, policy=policy, seed=fold_seed
                ),
            )
            e, m, ev, mv = [
                np.asarray(nuisance[key])
                for key in (
                    "training_propensity",
                    "training_outcome",
                    "validation_propensity",
                    "validation_outcome",
                )
            ]
            eligible_valid = linear.propensity_eligibility(
                ev, policy.min_propensity, policy.max_propensity
            )
            for i, row_id in enumerate(valid_ids):
                oof.append(
                    {
                        "_oci_row_id": int(row_id),
                        "treatment": float(tv[i]),
                        "outcome": float(yv[i]),
                        "propensity": float(ev[i]),
                        "outcome_prediction": float(mv[i]),
                        "effect_eligible": bool(eligible_valid[i]),
                    }
                )
            for repeat in range(cfg.repeats):
                cell_seed = fold_seed + repeat * 100
                rows = (
                    np.arange(len(train))
                    if repeat == 0
                    else _subsample(t, cfg.row_fraction, np.random.default_rng(cell_seed))
                )
                sampled = train.iloc[rows].reset_index(drop=True)
                ratio = float(cfg.l1_ratios[repeat % len(cfg.l1_ratios)])
                subsets = candidate_subsets(
                    list(by_id), repeat=repeat, subset_size=cfg.feature_subset_size, seed=cell_seed
                )
                for family in FAMILIES:
                    groups = (
                        subsets
                        if family in {"predictive_forest", "causal_forest"}
                        else [list(by_id)]
                    )
                    for subset_index, feature_ids in enumerate(groups):
                        leaf = f"fold_{fold:03d}/repeat_{repeat:03d}/{family}_{subset_index:03d}"

                        def compute():
                            selected_definitions = [by_id[k] for k in feature_ids]
                            nuisance_view = {"e": e[rows], "m": m[rows], "ev": ev, "mv": mv}
                            effect_family = family in {
                                "orthogonal_linear",
                                "univariable_rlearner",
                                "causal_forest",
                            }
                            keep = (
                                linear.propensity_eligibility(
                                    e[rows], policy.min_propensity, policy.max_propensity
                                )
                                if effect_family
                                else np.ones(len(rows), dtype=bool)
                            )
                            keepv = (
                                eligible_valid if effect_family else np.ones(len(valid), dtype=bool)
                            )
                            fit, hold = sampled.loc[keep].reset_index(drop=True), valid.loc[
                                keepv
                            ].reset_index(drop=True)
                            tt, yy, ttval, yyval = (
                                t[rows][keep],
                                y[rows][keep],
                                tv[keepv],
                                yv[keepv],
                            )
                            nv = {
                                "e": nuisance_view["e"][keep],
                                "m": nuisance_view["m"][keep],
                                "ev": ev[keepv],
                                "mv": mv[keepv],
                            }
                            try:
                                if len(fit) < 4 or len(hold) < 2:
                                    raise NotEstimable("insufficient_eligible_rows")
                                if effect_family and len(np.unique(tt)) < 2:
                                    raise NotEstimable("one_treatment_arm_in_effect_training_rows")
                                if family == "univariable":
                                    result = _univariable(
                                        fit,
                                        selected_definitions,
                                        tt,
                                        yy,
                                        binary=binary,
                                        policy=policy,
                                    )
                                elif family in {
                                    "penalized_main",
                                    "penalized_interactions",
                                    "orthogonal_linear",
                                }:
                                    result = _penalized_family(
                                        family,
                                        fit,
                                        hold,
                                        selected_definitions,
                                        tt,
                                        yy,
                                        ttval,
                                        yyval,
                                        binary=binary,
                                        policy=policy,
                                        seed=cell_seed,
                                        ratio=ratio,
                                        nuisance=nv,
                                    )
                                elif family == "univariable_rlearner":
                                    result = _univariable_rlearner(
                                        fit,
                                        hold,
                                        selected_definitions,
                                        tt - nv["e"],
                                        yy - nv["m"],
                                        ttval - nv["ev"],
                                        yyval - nv["mv"],
                                        policy=policy,
                                    )
                                else:
                                    result = _forests(
                                        family,
                                        fit,
                                        hold,
                                        selected_definitions,
                                        tt,
                                        yy,
                                        ttval,
                                        yyval,
                                        binary=binary,
                                        policy=policy,
                                        seed=cell_seed + subset_index,
                                        nuisance=nv,
                                    )
                            except NotEstimable as exc:
                                roles = (
                                    ("effect",)
                                    if effect_family
                                    else (
                                        ("treatment", "outcome")
                                        if family in {"predictive_forest", "penalized_main"}
                                        else ROLES
                                    )
                                )
                                result = {
                                    "records": [
                                        _record(k, r, status="not_evaluable")
                                        for k in feature_ids
                                        for r in roles
                                    ],
                                    "audit": {"status": str(exc)},
                                }
                            return {
                                "family": family,
                                "inner_fold": fold,
                                "repeat": repeat,
                                "candidate_ids": feature_ids,
                                "seed": cell_seed + subset_index,
                                "fit_row_ids": fit._oci_row_id.astype(int).tolist(),
                                "validation_row_ids": hold._oci_row_id.astype(int).tolist(),
                                **result,
                            }

                        LOGGER.info("Stage 2 multi-model %s", leaf)
                        cells.append(_checkpoint(directory, leaf, fingerprint, compute))
    summaries = aggregate_evidence(cells, list(by_id))
    availability = {
        family: sum(
            any(r["status"] == "ok" for r in c["records"]) for c in cells if c["family"] == family
        )
        for family in FAMILIES
    }
    if not any(availability.values()):
        raise NotEstimable("all_model_families_were_unevaluable")
    # Roles are assigned only after the LLM themes and reconciliation passes.
    decisions = [
        {
            "feature_id": _key(f),
            "roles": (
                list(f.get("roles", [])) if f.get("configured_explicit_feature") is True else []
            ),
            "selection_source": "awaiting_multi_model_adjudication",
        }
        for f in definitions
    ]
    locked = [dict(f) for f in definitions if f.get("configured_explicit_feature") is True]
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "temporal_scope": "pre_index_treatment",
        "policy": policy.public_dict(),
        "input_fingerprint": fingerprint,
        "inner_folds": len(inner_splits),
        "model_families": list(FAMILIES),
        "evaluable_cells_by_family": availability,
        "multi_model_evidence": summaries,
        "cells": cells,
        "cross_fitted_nuisance_models": {
            "model_family": "group_elastic_net",
            "predictions": sorted(oof, key=lambda r: r["_oci_row_id"]),
            "predictions_are_inner_fold_out_of_fold": True,
        },
        "decisions": decisions,
        "retained_feature_ids": [_key(f) for f in locked],
        "selection_authority": "multi_model",
        "latent_construction": "disabled",
        "stability_contract": {
            "support_is_conditional_on_evaluation": True,
            "repeats_share_patients_and_are_not_independent_experiments": True,
            "nuisances_fixed_across_resamples_within_inner_fold": True,
            "no_hard_p_value_frequency_or_top_n_gate": True,
        },
    }
    if directory is not None:
        _write_json(
            directory / "summary.json",
            {k: v for k, v in report.items() if k not in {"cells", "cross_fitted_nuisance_models"}},
        )
    return locked, report, definitions, []
