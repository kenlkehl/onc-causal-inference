"""Grouped binary DINA adapters for Stage 2 selection and final estimation."""

import logging

import numpy as np

from ..models import dina

LOGGER = logging.getLogger(__name__)


def _not_estimable(reason, diagnostics):
    from .stage2_multi_model_selection import NotEstimable

    error = NotEstimable(reason)
    error.diagnostics = diagnostics
    return error


def unpack_nuisances(value):
    if value.get("dina_nuisance_status") == "not_estimable":
        from .stage2_multi_model_selection import NotEstimable

        raise NotEstimable(value["dina_nuisance_reason"])
    return (
        dina.nuisances(value["training_propensity"], value["training_mu0"], value["training_mu1"]),
        dina.nuisances(
            value["validation_propensity"], value["validation_mu0"], value["validation_mu1"]
        ),
    )


def fit_effect(
    *,
    train,
    valid,
    definitions,
    modifier_ids,
    treatment,
    outcome,
    policy,
    seed,
    nuisance=None,
    ridge=None,
):
    """Fit group-selected log OR; return both native and probability scales.

    There are no validation outcomes or treatments in this interface. Every
    encoded level and missingness column belongs to its parent feature group.
    """
    from . import stage2_multi_model_selection as numerical
    from . import stage2_elastic_net_selection as linear

    treatment, outcome = np.asarray(treatment, float), np.asarray(outcome, float)
    if nuisance is None:
        nuisance = unpack_nuisances(
            numerical._nuisances(
                train,
                valid,
                definitions,
                treatment,
                outcome,
                binary=True,
                policy=policy,
                seed=seed + 100_000,
            )
        )
    fit_nu, valid_nu = nuisance
    selected_defs = [f for f in definitions if numerical._key(f) in set(modifier_ids)]

    def encode(fit, hold):
        return linear._winsorize_modifier_design(
            linear._encode_design(
                fit, hold, selected_defs, categorical_min_count=policy.categorical_min_count
            ),
            quantile=policy.modifier_continuous_winsor_quantile,
        )

    design = encode(train, valid)
    alphas = (
        [float(ridge) / max(len(train), 1)]
        if ridge is not None
        else np.logspace(
            policy.minimum_log10_alpha,
            policy.maximum_log10_alpha,
            policy.multi_model.regularization_grid_size,
        )[::-1]
    )
    ratio = 0.0 if ridge is not None else policy.l1_ratio
    cv_losses = []
    cv_diagnostics = []
    splits = linear._crossfit_indices(
        treatment, requested_folds=policy.internal_cv_folds, seed=seed
    )
    if ridge is None and not splits:
        raise numerical.NotEstimable("insufficient_rows_for_dina_penalty_cv")
    if ridge is None:
        for fold_index, (fit_rows, hold_rows) in enumerate(splits, start=1):
            encoded = encode(train.iloc[fit_rows], train.iloc[hold_rows])
            fold_losses = []
            initial = None  # Encoding and coefficients belong to this CV partition only.
            for alpha in alphas:
                try:
                    model = dina.fit(
                        encoded.train,
                        outcome[fit_rows],
                        treatment[fit_rows],
                        fit_nu["a"][fit_rows],
                        fit_nu["nu"][fit_rows],
                        groups=encoded.column_feature_ids,
                        regularization=float(alpha),
                        group_ratio=ratio,
                        max_iter=policy.max_iter,
                        tolerance=policy.optimization_tolerance,
                        initial=initial,
                    )
                except dina.DinaConvergenceError as exc:
                    # A failed trial must not abort other penalties, nor become
                    # a zero loss or a warm start. Invalid input still propagates.
                    fold_losses.append(np.inf)
                    cv_diagnostics.append({**exc.diagnostics, "cv_fold": fold_index,
                                           "regularization": float(alpha), "validation_loss": None})
                    LOGGER.warning("Stage 2 DINA penalty CV fold=%s: %s", fold_index, exc)
                    continue
                initial = model
                heldout_loss = float(dina.loss(
                    outcome[hold_rows], treatment[hold_rows], fit_nu["a"][hold_rows],
                    fit_nu["nu"][hold_rows], model.predict(encoded.valid),
                ).mean())
                fold_losses.append(heldout_loss if np.isfinite(heldout_loss) else np.inf)
                cv_diagnostics.append({**model.diagnostics, "cv_fold": fold_index,
                    "regularization": float(alpha),
                    "validation_loss": heldout_loss if np.isfinite(heldout_loss) else None})
            cv_losses.append(fold_losses)
    cv_means = np.mean(cv_losses, axis=0) if cv_losses else None
    if cv_means is not None and not np.isfinite(cv_means).any():
        raise _not_estimable("no_dina_regularization_converged_in_every_cv_fold",
                             {"cv_fits": cv_diagnostics})
    best = int(np.argmin(cv_means)) if cv_means is not None else 0
    if cv_losses and policy.modifier_one_standard_error_rule:
        values = np.asarray(cv_losses)
        threshold = values[:, best].mean() + values[:, best].std(ddof=1) / np.sqrt(len(values))
        best = int(np.flatnonzero(np.isfinite(cv_means) & (cv_means <= threshold))[0])
    try:
        # The full-training encoding can differ from every CV encoding. Start
        # fresh; never copy coefficients across that boundary or tune on validation.
        model = dina.fit(
            design.train,
            outcome,
            treatment,
            fit_nu["a"],
            fit_nu["nu"],
            groups=design.column_feature_ids,
            regularization=float(alphas[best]),
            group_ratio=ratio,
            max_iter=policy.max_iter,
            tolerance=policy.optimization_tolerance,
        )
    except dina.DinaConvergenceError as exc:
        raise _not_estimable("final_dina_fit_did_not_converge",
                             {"cv_fits": cv_diagnostics, "final_fit": exc.diagnostics}) from exc
    delta = model.predict(design.valid)
    mu0, mu1 = dina.counterfactuals(delta, valid_nu["a"], valid_nu["nu"])
    return {
        "tau": mu1 - mu0,
        "delta": delta,
        "mu0": mu0,
        "mu1": mu1,
        "model": model,
        "design": design,
        "nuisance": nuisance,
        "audit": {
            "model_family": "grouped_bernoulli_dina",
            "objective": dina.SCHEMA_VERSION,
            "outcome_model_contract": {
                "link": "logit",
                "effect_scale": "probability_difference",
                "fitted_effect_scale": "conditional_log_odds_ratio",
                "nuisance_centering": "bernoulli_variance_weighted",
            },
            "one_standard_error_rule": (
                bool(policy.modifier_one_standard_error_rule) if ridge is None else False
            ),
            "effect_scale": "conditional_log_odds_ratio",
            "prediction_scale": "probability_difference",
            "categorical_selection": "whole_feature_group",
            "regularization": float(alphas[best]),
            "l1_ratio": ratio,
            "cv_loss": float(np.mean(cv_losses, axis=0)[best]) if cv_losses else None,
            "selected_feature_ids": model.selected_groups(),
            "coefficient_feature_ids": list(design.column_feature_ids),
            "coefficients": model.coefficients.tolist(),
            "constant_log_odds_ratio": model.intercept,
            "fit_row_ids": train._oci_row_id.astype(int).tolist(),
            "validation_row_ids": valid._oci_row_id.astype(int).tolist(),
            "converged": model.converged,
            "iterations": model.iterations,
            "optimizer": model.diagnostics,
            "cv_fits": cv_diagnostics,
            "cate_intervals": "not_computed_for_grouped_dina",
        },
    }


def candidate_evidence(train, valid, definitions, t, y, tv, yv, *, policy, seed, nuisance):
    from . import stage2_multi_model_selection as numerical
    from . import stage2_evidence_stability as stability

    nt, nv = nuisance
    constant = dina.constant_effect(y, t, nt["a"], nt["nu"])
    baseline = float(dina.loss(yv, tv, nv["a"], nv["nu"], constant).mean())
    rows = []
    failed_fits = {}
    for feature in definitions:
        key = numerical._key(feature)
        try:
            result = fit_effect(
                train=train,
                valid=valid,
                definitions=definitions,
                modifier_ids=[key],
                treatment=t,
                outcome=y,
                policy=policy,
                seed=seed,
                nuisance=nuisance,
                ridge=policy.modifier_ridge_alpha,
            )
        except numerical.NotEstimable as exc:
            rows.append(numerical._record(key, "effect", status="not_evaluable"))
            failed_fits[key] = {"reason": str(exc), "diagnostics": getattr(exc, "diagnostics", {})}
            continue
        full = float(dina.loss(yv, tv, nv["a"], nv["nu"], result["delta"]).mean())
        rows.append(
            numerical._record(
                key,
                "effect",
                selected=baseline > full,
                score=baseline - full,
                loss_deltas=stability.pack(dina.loss(yv, tv, nv['a'], nv['nu'], constant)
                                          - dina.loss(yv, tv, nv['a'], nv['nu'], result['delta'])),
                effect_scale="conditional_log_odds_ratio",
            )
        )
    return {
        "records": rows,
        "audit": {
            "objective": dina.SCHEMA_VERSION,
            "baseline_heldout_dina_loss": baseline,
            "constant_log_odds_ratio": constant,
            "failed_candidate_fits": failed_fits,
        },
    }


def joint_evidence(
    train, valid, definitions, t, y, tv, yv, *, policy, seed, nuisance, permutation=False,
    measurement_groups=(),
):
    from . import stage2_multi_model_selection as numerical
    from . import stage2_evidence_stability as stability

    result = fit_effect(
        train=train,
        valid=valid,
        definitions=definitions,
        modifier_ids=[numerical._key(f) for f in definitions],
        treatment=t,
        outcome=y,
        policy=policy,
        seed=seed,
        nuisance=nuisance,
    )
    nt, nv = nuisance
    constant = dina.constant_effect(y, t, nt["a"], nt["nu"])

    def score(x):
        return dina.loss(yv, tv, nv["a"], nv["nu"], result["model"].predict(x))

    baseline = float(score(result["design"].valid).mean())
    gain = float(dina.loss(yv, tv, nv["a"], nv["nu"], constant).mean()) - baseline
    gains = {}
    if permutation:
        _, gains = stability.permutation_evidence(
            result["design"].valid,
            result["design"].column_feature_ids,
            score,
            repeats=policy.multi_model.permutation_repeats,
            seed=seed,
            groups=measurement_groups,
        )
    selected = set(result["model"].selected_groups())
    rows = []
    for f in definitions:
        key = numerical._key(f)
        ix = np.asarray(result["design"].column_feature_ids) == key
        importance = (
            gains.get(key, {}).get("gain", 0.0)
            if permutation
            else float(np.linalg.norm(result["model"].coefficients[ix]))
        )
        rows.append(
            numerical._record(
                key,
                "effect",
                selected=importance > 0 if permutation else key in selected,
                score=importance,
                model_gain=gain,
                **({'loss_deltas': gains[key]['loss_deltas'],
                    **stability.group_record(key, measurement_groups, gains)} if key in gains else {}),
                effect_scale="conditional_log_odds_ratio",
            )
        )
    return {
        "records": rows,
        "audit": {**result["audit"], "heldout_loss": baseline, "heldout_gain_over_constant": gain,
                  "model_loss_deltas": stability.pack(dina.loss(yv, tv, nv['a'], nv['nu'], constant)
                                                       - score(result['design'].valid))},
    }
