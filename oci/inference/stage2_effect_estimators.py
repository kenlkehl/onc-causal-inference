"""Shared train-only interaction outcome model for validation and final refits."""

from dataclasses import dataclass
from functools import partial

import numpy as np
from threadpoolctl import threadpool_limits

from . import stage2_elastic_net_selection as linear
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


# Separation fallback for the selected modifiers' unpenalized columns: the fixed
# ridge 0.5 * value * ||b||^2 on the mean-loss scale (standardized columns); for
# binary outcomes a N(0, 1 / (value * n)) prior on each coefficient. Not tuned.
SELECTED_MODIFIER_SEPARATION_RIDGE = 0.01
# An accepted unpenalized fit changes the linear predictor by at most this much
# across any selected-modifier interaction column's observed training range: log
# odds for binary outcomes, training-outcome standard deviations for continuous
# ones. Separation that moves the CATE always moves an interaction coefficient;
# main-effect columns are reported only, because their size also reflects
# collinearity with the prognostic adjustment covariate, penalized or not.
SELECTED_MODIFIER_MAX_RANGE_EFFECT = 10.0

SELECTED_MODIFIER_PENALTY_DESCRIPTION = (
    "Under modifier_selection.rule orthogonal_tests the modifiers were already chosen by "
    "calibrated tests, so the final interaction model does not shrink them again: their "
    "treatment-interaction columns and their main-effect columns (strong hierarchy) are "
    "exempt from the tuned group elastic net, which still penalizes the other main "
    "effects and is tuned by cross-validated outcome loss over those groups only. The "
    "exemption is used only when 2 x (encoded modifier columns) is at most the guard "
    "limit: min(events, non-events) / events_per_parameter among the effect-eligible "
    "training rows for binary outcomes, rows / events_per_parameter for continuous "
    "outcomes. A fit that does not converge or whose selected-modifier interaction "
    "columns change the linear predictor by more than max_range_effect across their "
    "training range (separation) is refitted with the fixed separation ridge on the "
    "selected-modifier columns; if that also fails, or the guard fails, the fully "
    "penalized model is used."
)


@dataclass(frozen=True)
class UnpenalizedSelectedModifiers:
    """Exempt the selected modifiers' main effects and interactions from the tuned penalty."""

    events_per_parameter: float
    separation_ridge: float = SELECTED_MODIFIER_SEPARATION_RIDGE
    max_range_effect: float = SELECTED_MODIFIER_MAX_RANGE_EFFECT

    def public_dict(self):
        return {
            "events_per_parameter": float(self.events_per_parameter),
            "separation_ridge": float(self.separation_ridge),
            "max_range_effect": float(self.max_range_effect),
        }


def selected_modifier_penalty(statistical_policy, outcome_type):
    """The final interaction model's selected-modifier policy; None keeps one penalty."""
    chosen = statistical_policy.multi_model.modifier_selection
    if not chosen.resolved_unpenalized_selected_interactions:
        return None
    return UnpenalizedSelectedModifiers(
        events_per_parameter=chosen.resolved_unpenalized_events_per_parameter(outcome_type)
    )


def selected_modifier_penalty_summary(audit):
    """Compact architecture-diagnostics view of a fit's selected-modifier penalty."""
    record = audit.get("selected_modifier_penalty")
    if record is None:
        return {
            "enabled": False,
            "path": "fully_penalized",
            "reason": "disabled_by_configuration",
        }
    guard = record["guard"]
    return {
        "enabled": True,
        "path": record["path"],
        "reason": record["reason"],
        "unpenalized_groups": list(record["unpenalized_groups"]),
        "fixed_ridge_groups": list(record["fixed_ridge_groups"]),
        "fixed_ridge": record["fixed_ridge"],
        "unpenalized_parameters": guard["unpenalized_parameters"],
        "parameter_limit": guard["parameter_limit"],
        "guard_passed": guard["passed"],
        "attempted_paths": [attempt["path"] for attempt in record["attempts"]],
    }


def _parameter_guard(outcome, *, binary, modifier_columns, events_per_parameter):
    rows = len(outcome)
    parameters = 2 * int(sum(modifier_columns.values()))
    if binary:
        events = int(np.sum(np.asarray(outcome) == 1))
        counts = {"events": events, "non_events": rows - events}
        information = min(counts.values())
        basis = "min_events_non_events"
    else:
        counts = {}
        information = rows
        basis = "eligible_rows"
    limit = information / float(events_per_parameter)
    return {
        "unpenalized_parameters": parameters,
        "parameter_count_rule": "2 x encoded columns of the selected modifiers",
        "modifier_encoded_columns": dict(sorted(modifier_columns.items())),
        "eligible_rows": rows,
        **counts,
        "information_basis": basis,
        "information": int(information),
        "events_per_parameter": float(events_per_parameter),
        "parameter_limit": float(limit),
        "passed": parameters <= limit,
    }


def _separation_check(model, treatment, modifier_ids, *, binary, outcome, settings):
    """Accept an exempt-modifier fit only when its selected-modifier terms stay bounded."""
    if model["state"] is None:
        return {"passed": True, "reason": "constant_target_or_empty_design"}
    design = model["design"]
    x, _, _, groups = numerical._matrix(
        design, treatment, "interactions", modifier_ids=modifier_ids,
        adjustment=model.get("adjustment"),
    )
    names = [
        "__treatment__",
        *(f"__{name}__" for name in (model["adjustment"].names if model.get("adjustment") else ())),
        *design.column_names,
        *(
            f"treatment:{name}"
            for name, key in zip(design.column_names, design.column_feature_ids)
            if key in modifier_ids
        ),
    ]
    coefficients = np.asarray(model["coefficients"], dtype=float)
    if not np.isfinite(coefficients).all() or not np.isfinite(model["state"].intercept):
        return {"passed": False, "reason": "nonfinite_coefficients"}
    scale = 1.0 if binary else float(np.std(np.asarray(outcome, dtype=float)))
    scale = scale if np.isfinite(scale) and scale > 1e-12 else 1.0

    def largest(block):
        selected = {f"{block}:{key}" for key in modifier_ids}
        columns = [index for index, group in enumerate(groups) if group in selected]
        if not columns:
            return 0.0, None
        effects = np.abs(coefficients[columns]) * np.ptp(x[:, columns], axis=0) / scale
        worst = int(np.argmax(effects))
        return float(effects[worst]), names[columns[worst]]

    interaction, interaction_column = largest("effect")
    main, main_column = largest("main")
    linear_predictor = x @ coefficients + model["state"].intercept
    passed = interaction <= float(settings.max_range_effect)
    return {
        "passed": bool(passed),
        "reason": None if passed else "selected_modifier_range_effect_exceeds_limit",
        "max_range_effect": interaction,
        "max_range_effect_column": interaction_column,
        "range_effect_scale": "log_odds" if binary else "training_outcome_sd",
        "max_range_effect_limit": float(settings.max_range_effect),
        "max_main_effect_range_effect": main,
        "max_main_effect_range_effect_column": main_column,
        "max_abs_training_linear_predictor": float(np.max(np.abs(linear_predictor))),
        "cv_nonconverged_fits": int(sum(
            not fit["converged"]
            for path in model["audit"].get("cv_optimizer_audit", [])
            for fit in path
        )),
    }


def _fit_selected_modifier_penalty(
    fit, *, train, valid, definitions, modifier_ids, treatment, outcome, binary, policy,
    path_submit, settings,
):
    """Exempt the selected modifiers if the guard allows; fall back on separation."""
    design = linear._encode_design(
        train, valid, definitions, categorical_min_count=policy.categorical_min_count
    )
    encoded = {
        key: int(sum(feature == key for feature in design.column_feature_ids))
        for key in modifier_ids
    }
    guard = _parameter_guard(
        outcome, binary=binary, modifier_columns=encoded,
        events_per_parameter=settings.events_per_parameter,
    )
    record = {
        "rule": "orthogonal_tests",
        "settings": settings.public_dict(),
        "guard": guard,
        "attempts": [],
        "description": SELECTED_MODIFIER_PENALTY_DESCRIPTION,
    }

    def penalized(reason):
        model = fit(path_submit=path_submit)
        record.update(
            path="fully_penalized", reason=reason, unpenalized_groups=[],
            fixed_ridge_groups=[], fixed_ridge=None,
        )
        return model, record

    if not modifier_ids:
        return penalized("no_selected_modifiers")
    if guard["unpenalized_parameters"] == 0:
        return penalized("selected_modifiers_have_no_encoded_columns")
    if not guard["passed"]:
        return penalized("events_per_parameter_guard")
    for path, ridge in (("unpenalized", None), ("separation_ridge", settings.separation_ridge)):
        try:
            model = fit(
                path_submit=(
                    partial(path_submit, variant=f"selected_{path}")
                    if path_submit is not None
                    else None
                ),
                unpenalized_modifier_ids=modifier_ids,
                modifier_ridge=ridge,
            )
        except numerical.NotEstimable as exc:
            record["attempts"].append({"path": path, "accepted": False, "reason": str(exc)})
            continue
        check = _separation_check(
            model, treatment, modifier_ids, binary=binary, outcome=outcome, settings=settings
        )
        accepted = check.pop("passed")
        record["attempts"].append({"path": path, "accepted": accepted, **check})
        if accepted:
            # A constant-target model (no solver state) has no modifier groups.
            groups = list(model["audit"].get("selected_modifier_groups", {}).get("groups", []))
            record.update(
                path=path,
                reason=None if ridge is None else "unpenalized_fit_separated_or_did_not_converge",
                unpenalized_groups=groups if ridge is None else [],
                fixed_ridge_groups=[] if ridge is None else groups,
                fixed_ridge=None if ridge is None else float(ridge),
            )
            return model, record
    return penalized("separation_or_nonconvergence")


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


def ensemble_estimators(outcome_type):
    """Every final estimator available for the outcome type, in canonical order."""
    if outcome_type == "binary":
        return ("causal_forest", "linear_interactions", "dina")
    if outcome_type == "continuous":
        return ("causal_forest", "linear_interactions")
    raise ValueError("the estimator ensemble requires a binary or continuous outcome")


def effect_model_family(estimator, outcome_type):
    if estimator == "ensemble":
        ensemble_estimators(outcome_type)
        return "unweighted_estimator_ensemble"
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
    unpenalized_selected=None,
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

    ``unpenalized_selected`` (``UnpenalizedSelectedModifiers``; final estimation
    under the orthogonal_tests rule only) exempts the modifiers' main-effect and
    interaction columns from the tuned penalty, subject to the events-per-
    parameter guard and the separation fallback; the audit then records
    ``selected_modifier_penalty``. None keeps one tuned penalty for every main
    effect and interaction, as in every validation fit.
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
        if unpenalized_selected is None:
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
            penalty_record = None
        else:
            model, penalty_record = _fit_selected_modifier_penalty(
                partial(
                    numerical._fit_linear,
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
                    adjustment=adjustment,
                ),
                train=train,
                valid=valid,
                definitions=definitions,
                modifier_ids=modifier_ids,
                treatment=treatment,
                outcome=outcome,
                binary=binary,
                policy=policy,
                path_submit=path_submit,
                settings=unpenalized_selected,
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
            **(
                {"selected_modifier_penalty": penalty_record}
                if penalty_record is not None
                else {}
            ),
        },
    }
