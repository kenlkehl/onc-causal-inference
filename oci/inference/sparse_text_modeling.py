"""Shared sparse-text modeling primitives for active Stage 1 architectures."""

from __future__ import annotations

import logging
from dataclasses import asdict, fields
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold

from ..config import BoWViewConfig, TfidfVectorizerScientificConfig

LOGGER = logging.getLogger(__name__)

_BOW_VECTORIZER_PARAMETER_KEYS = frozenset(
    field.name for field in fields(TfidfVectorizerScientificConfig)
)
_BOW_MODEL_PARAMETER_KEYS = frozenset(
    {
        "bow_model",
        "logistic_c",
        "logistic_max_iter",
        "ridge_alpha",
        "logistic_scientific",
        "ridge_scientific",
        "forest_scientific",
        "xgboost_scientific",
        "empty_vocabulary_policy",
        "unsupported_sample_weight_policy",
    }
)


def _finite_or_none(value: Any) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _require_exact_constructor_mapping(
    params: Dict[str, Any],
    *,
    expected: frozenset[str],
    label: str,
) -> Dict[str, Any]:
    if not isinstance(params, dict):
        raise TypeError(f"{label} parameters must be a plain mapping")
    missing = sorted(expected - set(params))
    extra = sorted(set(params) - expected)
    if missing or extra:
        raise ValueError(
            f"{label} parameter mapping is not closed; missing={missing}, extra={extra}"
        )
    return params


def bow_vectorizer_params(view: BoWViewConfig) -> Dict[str, Any]:
    if type(view) is not BoWViewConfig:
        raise TypeError("BoW vectorizer parameters require BoWViewConfig")
    return asdict(view.vectorizer_scientific)


def bow_model_params(view: BoWViewConfig) -> Dict[str, Any]:
    if type(view) is not BoWViewConfig:
        raise TypeError("BoW model parameters require BoWViewConfig")
    return {
        "bow_model": str(view.bow_model),
        "logistic_c": float(view.logistic_c),
        "logistic_max_iter": int(view.logistic_max_iter),
        "ridge_alpha": float(view.ridge_alpha),
        "logistic_scientific": asdict(view.logistic_scientific),
        "ridge_scientific": asdict(view.ridge_scientific),
        "forest_scientific": asdict(view.forest_scientific),
        "xgboost_scientific": asdict(view.xgboost_scientific),
        "empty_vocabulary_policy": str(view.empty_vocabulary_policy),
        "unsupported_sample_weight_policy": str(view.unsupported_sample_weight_policy),
    }


def make_bow_vectorizer(params: Dict[str, Any]) -> TfidfVectorizer:
    params = _require_exact_constructor_mapping(
        params,
        expected=_BOW_VECTORIZER_PARAMETER_KEYS,
        label="BoW TF-IDF vectorizer",
    )
    dtype = {"float32": np.float32, "float64": np.float64}[str(params["dtype"])]
    return TfidfVectorizer(
        input="content",
        encoding=str(params["encoding"]),
        decode_error=str(params["decode_error"]),
        strip_accents=params["strip_accents"],
        lowercase=bool(params["lowercase"]),
        preprocessor=None,
        tokenizer=None,
        analyzer=str(params["analyzer"]),
        stop_words=params["stop_words"],
        token_pattern=params["token_pattern"],
        ngram_range=(int(params["ngram_range_min"]), int(params["ngram_range_max"])),
        min_df=int(params["min_df"]),
        max_df=float(params["max_df"]),
        max_features=(None if params["max_features"] is None else int(params["max_features"])),
        vocabulary=None,
        binary=bool(params["binary"]),
        dtype=dtype,
        norm=params["norm"],
        use_idf=bool(params["use_idf"]),
        smooth_idf=bool(params["smooth_idf"]),
        sublinear_tf=bool(params["sublinear_tf"]),
    )


def _forest_kwargs(config: MappingLike, *, classifier: bool, random_state: int) -> dict[str, Any]:
    return {
        "n_estimators": int(config["n_estimators"]),
        "criterion": str(
            config["classifier_criterion"] if classifier else config["regressor_criterion"]
        ),
        "max_depth": config["max_depth"],
        "min_samples_split": config["min_samples_split"],
        "min_samples_leaf": config["min_samples_leaf"],
        "min_weight_fraction_leaf": float(config["min_weight_fraction_leaf"]),
        "max_features": config["max_features"],
        "max_leaf_nodes": config["max_leaf_nodes"],
        "min_impurity_decrease": float(config["min_impurity_decrease"]),
        "oob_score": bool(config["oob_score"]),
        "n_jobs": 1,
        "random_state": random_state,
        "verbose": 0,
        "warm_start": bool(config["warm_start"]),
        "ccp_alpha": float(config["ccp_alpha"]),
        "max_samples": config["max_samples"],
        "monotonic_cst": config["monotonic_cst"],
    }


MappingLike = Dict[str, Any]


def make_bow_classifier(params: Dict[str, Any], *, random_state: int = 17):
    params = _require_exact_constructor_mapping(
        params,
        expected=_BOW_MODEL_PARAMETER_KEYS,
        label="BoW classifier",
    )
    model_name = str(params["bow_model"]).strip().lower()
    if model_name == "linear":
        config = dict(params["logistic_scientific"])
        return LogisticRegression(
            penalty=config["penalty"],
            dual=bool(config["dual"]),
            tol=float(config["tol"]),
            C=float(params["logistic_c"]),
            fit_intercept=bool(config["fit_intercept"]),
            intercept_scaling=float(config["intercept_scaling"]),
            class_weight=config["class_weight"],
            solver=str(config["solver"]),
            max_iter=int(params["logistic_max_iter"]),
            verbose=0,
            warm_start=bool(config["warm_start"]),
            n_jobs=1,
            l1_ratio=config["l1_ratio"],
            random_state=random_state,
        )
    config = dict(params["forest_scientific"])
    kwargs = _forest_kwargs(config, classifier=True, random_state=random_state)
    kwargs["class_weight"] = config["class_weight"]
    if model_name == "extratrees":
        return ExtraTreesClassifier(
            **kwargs,
            bootstrap=bool(config["extra_trees_bootstrap"]),
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            **kwargs,
            bootstrap=bool(config["random_forest_bootstrap"]),
        )
    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "bow_model='xgboost' requires xgboost; refusing learner substitution"
            ) from exc
        config = dict(params["xgboost_scientific"])
        return XGBClassifier(
            n_estimators=int(config["n_estimators"]),
            max_depth=int(config["max_depth"]),
            max_leaves=int(config["max_leaves"]),
            max_bin=int(config["max_bin"]),
            grow_policy=str(config["grow_policy"]),
            learning_rate=float(config["learning_rate"]),
            booster=str(config["booster"]),
            tree_method=str(config["tree_method"]),
            gamma=float(config["gamma"]),
            min_child_weight=float(config["min_child_weight"]),
            max_delta_step=float(config["max_delta_step"]),
            subsample=float(config["subsample"]),
            sampling_method=str(config["sampling_method"]),
            colsample_bytree=float(config["colsample_bytree"]),
            colsample_bylevel=float(config["colsample_bylevel"]),
            colsample_bynode=float(config["colsample_bynode"]),
            reg_alpha=float(config["reg_alpha"]),
            reg_lambda=float(config["reg_lambda"]),
            scale_pos_weight=float(config["scale_pos_weight"]),
            base_score=float(config["base_score"]),
            missing=np.nan,
            num_parallel_tree=int(config["num_parallel_tree"]),
            monotone_constraints=config["monotone_constraints"],
            interaction_constraints=config["interaction_constraints"],
            enable_categorical=bool(config["enable_categorical"]),
            max_cat_to_onehot=int(config["max_cat_to_onehot"]),
            max_cat_threshold=int(config["max_cat_threshold"]),
            multi_strategy=str(config["multi_strategy"]),
            objective=str(config["classifier_objective"]),
            eval_metric=str(config["classifier_eval_metric"]),
            random_state=random_state,
            n_jobs=1,
        )
    raise ValueError(f"Unsupported bow_model: {model_name}")


def make_bow_regressor(params: Dict[str, Any], *, random_state: int = 17):
    params = _require_exact_constructor_mapping(
        params,
        expected=_BOW_MODEL_PARAMETER_KEYS,
        label="BoW regressor",
    )
    model_name = str(params["bow_model"]).strip().lower()
    if model_name == "linear":
        config = dict(params["ridge_scientific"])
        return Ridge(
            alpha=float(params["ridge_alpha"]),
            fit_intercept=bool(config["fit_intercept"]),
            copy_X=True,
            max_iter=config["max_iter"],
            tol=float(config["tol"]),
            solver=str(config["solver"]),
            positive=bool(config["positive"]),
            random_state=random_state,
        )
    config = dict(params["forest_scientific"])
    kwargs = _forest_kwargs(config, classifier=False, random_state=random_state)
    if model_name == "extratrees":
        return ExtraTreesRegressor(
            **kwargs,
            bootstrap=bool(config["extra_trees_bootstrap"]),
        )
    if model_name == "random_forest":
        return RandomForestRegressor(
            **kwargs,
            bootstrap=bool(config["random_forest_bootstrap"]),
        )
    if model_name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "bow_model='xgboost' requires xgboost; refusing learner substitution"
            ) from exc
        config = dict(params["xgboost_scientific"])
        return XGBRegressor(
            n_estimators=int(config["n_estimators"]),
            max_depth=int(config["max_depth"]),
            max_leaves=int(config["max_leaves"]),
            max_bin=int(config["max_bin"]),
            grow_policy=str(config["grow_policy"]),
            learning_rate=float(config["learning_rate"]),
            booster=str(config["booster"]),
            tree_method=str(config["tree_method"]),
            gamma=float(config["gamma"]),
            min_child_weight=float(config["min_child_weight"]),
            max_delta_step=float(config["max_delta_step"]),
            subsample=float(config["subsample"]),
            sampling_method=str(config["sampling_method"]),
            colsample_bytree=float(config["colsample_bytree"]),
            colsample_bylevel=float(config["colsample_bylevel"]),
            colsample_bynode=float(config["colsample_bynode"]),
            reg_alpha=float(config["reg_alpha"]),
            reg_lambda=float(config["reg_lambda"]),
            base_score=float(config["base_score"]),
            missing=np.nan,
            num_parallel_tree=int(config["num_parallel_tree"]),
            monotone_constraints=config["monotone_constraints"],
            interaction_constraints=config["interaction_constraints"],
            enable_categorical=bool(config["enable_categorical"]),
            max_cat_to_onehot=int(config["max_cat_to_onehot"]),
            max_cat_threshold=int(config["max_cat_threshold"]),
            multi_strategy=str(config["multi_strategy"]),
            objective=str(config["regressor_objective"]),
            eval_metric=str(config["regressor_eval_metric"]),
            random_state=random_state,
            n_jobs=1,
        )
    raise ValueError(f"Unsupported bow_model: {model_name}")


def fit_regressor(
    model: Any,
    x: Any,
    y: np.ndarray,
    *,
    sample_weight: Optional[np.ndarray] = None,
    unsupported_sample_weight_policy: str,
) -> Any:
    if unsupported_sample_weight_policy not in {
        "fail_closed",
        "unweighted_legacy_compatibility",
    }:
        raise ValueError("unsupported sample-weight policy")
    if sample_weight is None:
        return model.fit(x, y)
    weights = np.asarray(sample_weight, dtype=float)
    if weights.shape[0] != len(y):
        raise ValueError("sample_weight must have one value per training row")
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    if float(np.sum(weights)) <= 0.0:
        if unsupported_sample_weight_policy == "fail_closed":
            raise ValueError("sample_weight has no positive finite mass")
        return model.fit(x, y)
    try:
        return model.fit(x, y, sample_weight=weights)
    except TypeError as exc:
        if unsupported_sample_weight_policy == "fail_closed":
            raise TypeError(
                f"BoW regressor {type(model).__name__} does not accept configured sample weights"
            ) from exc
        LOGGER.warning(
            "Legacy BoW compatibility: %s does not accept sample_weight; fitting unweighted",
            type(model).__name__,
        )
        return model.fit(x, y)


def bounded_fold_count(requested: int, n_rows: int) -> int:
    if n_rows < 2:
        raise ValueError("At least two rows are required for cross-fitting")
    return max(2, min(int(requested), int(n_rows)))


def select_ridge_alpha(
    ridge_alphas: Sequence[float],
    heldout_losses: Sequence[float],
    *,
    criterion: str,
) -> Dict[str, Any]:
    """Pick the penalty with the lowest held-out loss; ties go to the larger penalty.

    One penalty is recorded as ``fixed``. The record lists the grid ascending.
    """
    alphas = np.asarray(ridge_alphas, dtype=float)
    losses = np.asarray(heldout_losses, dtype=float)
    if alphas.ndim != 1 or not len(alphas) or alphas.shape != losses.shape:
        raise ValueError("one held-out loss is required per ridge penalty")
    if not np.isfinite(losses).any():
        raise ValueError("no ridge penalty has a finite held-out loss")
    descending = np.argsort(-alphas, kind="stable")
    ranked = np.where(np.isfinite(losses[descending]), losses[descending], np.inf)
    best = int(descending[int(np.argmin(ranked))])
    ascending = np.argsort(alphas, kind="stable")
    return {
        "selection": "cross_validated" if len(alphas) > 1 else "fixed",
        "criterion": str(criterion),
        "ridge_alpha": float(alphas[best]),
        "at_grid_edge": bool(len(alphas) > 1 and best in (ascending[0], ascending[-1])),
        "ridge_alphas": [float(value) for value in alphas[ascending]],
        "heldout_loss": [_finite_or_none(value) for value in losses[ascending]],
    }


def binary_split_items(
    labels: np.ndarray,
    *,
    requested_folds: int,
    random_state: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    values, counts = np.unique(labels.astype(int), return_counts=True)
    if len(values) >= 2 and int(np.min(counts)) >= 2:
        folds = max(2, min(int(requested_folds), int(np.min(counts)), len(labels)))
        splitter = StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=random_state,
        )
        return [
            (np.asarray(fit), np.asarray(heldout))
            for fit, heldout in splitter.split(np.zeros(len(labels)), labels)
        ]
    splitter = KFold(
        n_splits=bounded_fold_count(requested_folds, len(labels)),
        shuffle=True,
        random_state=random_state,
    )
    return [
        (np.asarray(fit), np.asarray(heldout))
        for fit, heldout in splitter.split(np.zeros(len(labels)))
    ]


def model_feature_scores(model: Any, n_features: int) -> np.ndarray:
    coef = getattr(model, "coef_", None)
    if coef is not None:
        values = np.asarray(coef, dtype=float).ravel()
    else:
        importances = getattr(model, "feature_importances_", None)
        if importances is not None:
            values = np.asarray(importances, dtype=float).ravel()
        else:
            booster = getattr(model, "get_booster", None)
            if booster is None:
                return np.zeros(n_features, dtype=float)
            try:
                score = booster().get_score(importance_type="gain")
            except Exception:
                return np.zeros(n_features, dtype=float)
            values = np.zeros(n_features, dtype=float)
            for key, value in score.items():
                if str(key).startswith("f") and 0 <= int(str(key)[1:]) < n_features:
                    values[int(str(key)[1:])] = float(value)
            return values
    resized = np.zeros(n_features, dtype=float)
    resized[: min(n_features, len(values))] = values[: min(n_features, len(values))]
    return resized


def model_feature_scores_are_signed(model: Any) -> bool:
    """Return whether :func:`model_feature_scores` yields signed coefficients.

    Linear models (and DINA) expose a signed ``coef_``. Tree ensembles expose
    unsigned impurity or gain importances: their lowest values are merely the
    least important terms and carry no negative direction, so callers must not
    publish a ``*_negative`` list for them.
    """

    return getattr(model, "coef_", None) is not None


def top_feature_rows(
    features: np.ndarray,
    scores: np.ndarray,
    top_n: int,
    *,
    descending: bool = True,
    treatment_coef: Optional[np.ndarray] = None,
    outcome_coef: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    if len(features) == 0:
        return []
    order = np.argsort(scores)
    if descending:
        order = order[::-1]
    rows: List[Dict[str, Any]] = []
    for index in order[:top_n]:
        row: Dict[str, Any] = {
            "feature": str(features[index]),
            "score": _finite_or_none(scores[index]),
        }
        if treatment_coef is not None:
            row["treatment_score"] = _finite_or_none(treatment_coef[index])
            row["abs_treatment_score"] = _finite_or_none(abs(treatment_coef[index]))
        if outcome_coef is not None:
            row["outcome_score"] = _finite_or_none(outcome_coef[index])
            row["abs_outcome_score"] = _finite_or_none(abs(outcome_coef[index]))
        rows.append(row)
    return rows


# Compatibility aliases let migrated callers retain their established local names.
_bow_model_params = bow_model_params
_bow_vectorizer_params = bow_vectorizer_params
_make_bow_classifier = make_bow_classifier
_make_bow_regressor = make_bow_regressor
_make_bow_vectorizer = make_bow_vectorizer
_fit_regressor = fit_regressor
_bounded_fold_count = bounded_fold_count
_select_ridge_alpha = select_ridge_alpha
_binary_split_items = binary_split_items
_model_feature_scores = model_feature_scores
_model_feature_scores_are_signed = model_feature_scores_are_signed
_top_feature_rows = top_feature_rows


__all__ = [
    "binary_split_items",
    "bow_model_params",
    "bow_vectorizer_params",
    "bounded_fold_count",
    "fit_regressor",
    "make_bow_classifier",
    "make_bow_regressor",
    "make_bow_vectorizer",
    "model_feature_scores",
    "model_feature_scores_are_signed",
    "select_ridge_alpha",
    "top_feature_rows",
]
