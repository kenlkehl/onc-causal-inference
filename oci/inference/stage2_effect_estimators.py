"""Shared train-only interaction outcome model for validation and final refits."""

import numpy as np
from threadpoolctl import threadpool_limits

from . import stage2_multi_model_selection as numerical

# Probabilities are bounded away from 0/1 before taking logits so one extreme
# nuisance prediction cannot dominate an unpenalized covariate or a forest split.
NUISANCE_LOGIT_CLIP = 1e-4

NUISANCE_ADJUSTMENT_DESCRIPTION = (
    "Unpenalized adjustment covariates logit(e_hat) and the prognostic score "
    "(logit(mu0_hat) for binary outcomes, mu0_hat for continuous outcomes) were added "
    "to the penalized interaction outcome model to prevent regularization-induced "
    "confounding (Hahn et al. 2018): shrinking many confounder main effects while T "
    "is unpenalized pushes confounding bias into T and T:modifier coefficients. "
    "Training rows use cross-fitted out-of-fold nuisances; validation/held-out rows "
    "use nuisances fitted on the training rows only. The covariates are never "
    "interacted with T."
)

PROGNOSTIC_FOREST_DESCRIPTION = (
    "For binary outcomes the causal forest's effect-modifier matrix X includes the "
    "cross-fitted prognostic score logit(mu0_hat). On the risk-difference scale the "
    "CATE varies with baseline risk even under a constant odds ratio, which a forest "
    "restricted to selected modifiers cannot represent. Training rows use out-of-fold "
    "mu0_hat; held-out/validation rows use mu0_hat fitted on the training rows only."
)


def bounded_logit(probability):
    p = np.clip(np.asarray(probability, dtype=float), NUISANCE_LOGIT_CLIP, 1 - NUISANCE_LOGIT_CLIP)
    return np.log(p / (1.0 - p))


def prognostic_score(mu0, *, binary):
    """Prognostic score on the outcome model's linear-predictor scale."""
    values = np.asarray(mu0, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("prognostic score requires finite mu0 predictions")
    return bounded_logit(values) if binary else values


def nuisance_adjustment(*, train_propensity, valid_propensity, train_mu0, valid_mu0, binary):
    """Build the unpenalized interaction-model covariates from honest nuisances.

    Training values must be out-of-fold for the training rows; validation values
    must come from models fitted on training rows only.
    """
    columns = []
    for train_values, valid_values in (
        (bounded_logit(train_propensity), bounded_logit(valid_propensity)),
        (prognostic_score(train_mu0, binary=binary), prognostic_score(valid_mu0, binary=binary)),
    ):
        columns.append((np.asarray(train_values, float), np.asarray(valid_values, float)))
    names = ["logit_propensity", "logit_prognostic_mu0" if binary else "prognostic_mu0"]
    return {
        "train": np.column_stack([c[0] for c in columns]),
        "valid": np.column_stack([c[1] for c in columns]),
        "names": names,
    }


def effect_model_family(estimator, outcome_type):
    if estimator == "dina":
        if outcome_type != "binary":
            raise ValueError("DINA final architecture requires binary outcomes")
        return "grouped_bernoulli_dina"
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
    *,
    train,
    valid,
    definitions,
    modifier_ids,
    treatment,
    outcome,
    binary,
    policy,
    seed,
    path_submit=None,
    adjustment=None,
):
    """Fit g(E[Y]) = intercept + T + A + main effects + T:selected modifiers.

    Group elastic-net penalties are tuned by outcome loss inside training only.
    The link is logistic for binary outcomes, identity otherwise. Validation
    labels/treatments are deliberately absent from this API. Main effects remain
    in the design even when their interaction is not selected. The intercept and
    main treatment coefficient are exempt from both group-lasso and ridge
    penalties in every tuning fit and the final refit.

    ``adjustment`` (from ``nuisance_adjustment``) adds the cross-fitted
    logit(e_hat) and prognostic score as further UNPENALIZED covariates A.
    Regularization-induced confounding (Hahn et al. 2018, Bayesian Analysis):
    when dozens of confounder main effects are shrunk toward zero but T is not,
    the confounding those shrunken coefficients should absorb flows into the
    unpenalized T coefficient and the T:modifier terms, biasing the ATE and
    CATEs. An unpenalized propensity (and prognostic) covariate carries that
    confounding instead, whatever the penalty does to the individual columns.
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
            path_submit=path_submit,
            adjustment=adjustment,
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
                adjustment=model.get("adjustment"),
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
            "model_family": (
                "penalized_logistic_interactions" if binary else "penalized_linear_interactions"
            ),
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
            "nuisance_adjustment": (
                {
                    "columns": list(model["adjustment"].names)
                    if model.get("adjustment") is not None
                    else [],
                    "requested_columns": list(map(str, adjustment["names"])),
                    "penalized": False,
                    "interacted_with_treatment": False,
                    "description": NUISANCE_ADJUSTMENT_DESCRIPTION,
                }
                if adjustment is not None
                else {"columns": [], "description": "no nuisance adjustment covariates supplied"}
            ),
        },
    }
