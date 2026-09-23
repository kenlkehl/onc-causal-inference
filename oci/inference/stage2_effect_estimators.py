"""Shared train-only interaction outcome model for validation and final refits."""

import numpy as np
from threadpoolctl import threadpool_limits

from . import stage2_multi_model_selection as numerical


def effect_model_family(estimator, outcome_type):
    if estimator == "causal_forest":
        return "causal_forest_dml"
    if estimator == "linear_interactions":
        return (
            "penalized_logistic_interactions"
            if outcome_type == "binary"
            else "penalized_linear_interactions"
        )
    raise ValueError(f"unsupported final effect estimator: {estimator}")


def fit_interaction_effect(
    *, train, valid, definitions, modifier_ids, treatment, outcome, binary, policy, seed
):
    """Fit g(E[Y]) = intercept + T + main effects + T:selected modifiers.

    Group elastic-net penalties are tuned by outcome loss inside training only.
    The link is logistic for binary outcomes, identity otherwise. Validation
    labels/treatments are deliberately absent from this API. Main effects remain
    in the design even when their interaction is not selected.
    """
    modifier_ids = set(modifier_ids)
    if not modifier_ids.issubset({numerical._key(f) for f in definitions}):
        raise ValueError("interaction modifiers must have main-effect definitions")
    treatment, outcome = np.asarray(treatment, float), np.asarray(outcome, float)
    if len(train) < 4 or len(np.unique(treatment)) < 2:
        raise numerical.NotEstimable("insufficient_interaction_training_rows_or_treatment_arms")
    # These small, repeated matrix fits are slower with large BLAS thread pools;
    # the production adapter can also run several outer folds concurrently.
    with threadpool_limits(limits=1):
        model = numerical._fit_linear(
            train,
            valid,
            definitions,
            outcome,
            binary=binary,
            kind="interactions",
            treatment=treatment,
            valid_treatment=np.zeros(len(valid)),
            modifier_ids=modifier_ids,
            policy=policy,
            seed=seed,
            ratio=policy.l1_ratio,
        )
    mu = []
    for arm in (0, 1):
        if model["state"] is None:
            prediction = model["prediction"].copy()
        else:
            design, _, _, _ = numerical._matrix(
                model["design"],
                np.full(len(valid), arm),
                "interactions",
                valid=True,
                modifier_ids=modifier_ids,
            )
            prediction = numerical.linear._state_prediction(model["state"], design, binary=binary)
        mu.append(np.asarray(prediction, float))
    tau = mu[1] - mu[0]
    if not np.isfinite(tau).all():
        raise ValueError("interaction model produced nonfinite effects")
    return {
        "tau": tau,
        "mu0": mu[0],
        "mu1": mu[1],
        "audit": {
            **model["audit"],
            "model_family": "penalized_logistic_interactions"
            if binary
            else "penalized_linear_interactions",
            "outcome_model_contract": {
                "link": "logit" if binary else "identity",
                "effect_scale": "probability_difference" if binary else "outcome_difference",
            },
            "main_feature_ids": [numerical._key(f) for f in definitions],
            "modifier_feature_ids": sorted(modifier_ids),
            "encoded_columns": len(model["coefficients"]),
            "coefficient_feature_ids": model["keys"],
            "coefficient_blocks": model["blocks"],
            "coefficients": model["coefficients"].tolist(),
            "fit_row_ids": train._oci_row_id.astype(int).tolist(),
            "validation_row_ids": valid._oci_row_id.astype(int).tolist(),
            "cate_intervals": "not_computed_for_penalized_interaction_model",
        },
    }
