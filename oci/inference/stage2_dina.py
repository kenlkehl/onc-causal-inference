"""Grouped binary DINA adapters for Stage 2 selection and final estimation."""

import numpy as np

from ..models import dina


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
            (
                policy.multi_model.regularization_grid_size
                if policy.selection_mode == "multi_model"
                else policy.regularization_grid_size
            ),
        )[::-1]
    )
    ratio = 0.0 if ridge is not None else policy.l1_ratio
    cv_losses = []
    splits = linear._crossfit_indices(
        treatment, requested_folds=policy.internal_cv_folds, seed=seed
    )
    if ridge is None and not splits:
        raise numerical.NotEstimable("insufficient_rows_for_dina_penalty_cv")
    if ridge is None:
        for fit_rows, hold_rows in splits:
            encoded = encode(train.iloc[fit_rows], train.iloc[hold_rows])
            fold_losses = []
            for alpha in alphas:
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
                )
                fold_losses.append(
                    float(
                        dina.loss(
                            outcome[hold_rows],
                            treatment[hold_rows],
                            fit_nu["a"][hold_rows],
                            fit_nu["nu"][hold_rows],
                            model.predict(encoded.valid),
                        ).mean()
                    )
                )
            cv_losses.append(fold_losses)
    best = int(np.argmin(np.mean(cv_losses, axis=0))) if cv_losses else 0
    if cv_losses and policy.modifier_one_standard_error_rule:
        values = np.asarray(cv_losses)
        threshold = values[:, best].mean() + values[:, best].std(ddof=1) / np.sqrt(len(values))
        best = int(np.flatnonzero(values.mean(axis=0) <= threshold)[0])
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
            "cate_intervals": "not_computed_for_grouped_dina",
        },
    }


def candidate_evidence(train, valid, definitions, t, y, tv, yv, *, policy, seed, nuisance):
    from . import stage2_multi_model_selection as numerical

    nt, nv = nuisance
    constant = dina.constant_effect(y, t, nt["a"], nt["nu"])
    baseline = float(dina.loss(yv, tv, nv["a"], nv["nu"], constant).mean())
    rows = []
    for feature in definitions:
        key = numerical._key(feature)
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
        full = float(dina.loss(yv, tv, nv["a"], nv["nu"], result["delta"]).mean())
        rows.append(
            numerical._record(
                key,
                "effect",
                selected=baseline > full,
                score=baseline - full,
                effect_scale="conditional_log_odds_ratio",
            )
        )
    return {
        "records": rows,
        "audit": {
            "objective": dina.SCHEMA_VERSION,
            "baseline_heldout_dina_loss": baseline,
            "constant_log_odds_ratio": constant,
        },
    }


def joint_evidence(
    train, valid, definitions, t, y, tv, yv, *, policy, seed, nuisance, permutation=False
):
    from . import stage2_multi_model_selection as numerical

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
        return float(dina.loss(yv, tv, nv["a"], nv["nu"], result["model"].predict(x)).mean())

    baseline = score(result["design"].valid)
    gain = float(dina.loss(yv, tv, nv["a"], nv["nu"], constant).mean()) - baseline
    gains = {}
    if permutation:
        _, gains = numerical.grouped_permutation_gain(
            result["design"].valid,
            result["design"].column_feature_ids,
            score,
            repeats=policy.multi_model.permutation_repeats,
            seed=seed,
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
                effect_scale="conditional_log_odds_ratio",
            )
        )
    return {
        "records": rows,
        "audit": {**result["audit"], "heldout_loss": baseline, "heldout_gain_over_constant": gain},
    }


def legacy_screen(
    *,
    train,
    valid,
    definitions,
    treatment,
    outcome,
    valid_treatment,
    valid_outcome,
    nuisance,
    policy,
    seed,
    candidate=False,
):
    """Adapt older selection callers; report labels are normalized at their boundary."""
    from . import stage2_multi_model_selection as numerical
    from .stage2_elastic_net_selection import propensity_eligibility

    if isinstance(nuisance, dict) and nuisance.get("status") == "not_evaluable":
        return {
            **nuisance,
            "selected_feature_ids": [],
            "feature_group_l2_norms": {},
            "heldout_reduced_r_loss": None,
            "heldout_full_r_loss": None,
            "heldout_r_loss_improvement": None,
            "heldout_relative_r_loss_improvement": None,
            **(
                {"feature_id": numerical._key(definitions[0]), "name": definitions[0]["name"]}
                if candidate
                else {}
            ),
        }
    if nuisance is None:
        nuisance = unpack_nuisances(
            numerical._nuisances(
                train, valid, definitions, treatment, outcome, binary=True, policy=policy, seed=seed
            )
        )
    nt, nv = nuisance
    keep = propensity_eligibility(nt["e"], policy.min_propensity, policy.max_propensity)
    keepv = propensity_eligibility(nv["e"], policy.min_propensity, policy.max_propensity)
    nt, nv = ({k: v[keep] for k, v in nt.items()}, {k: v[keepv] for k, v in nv.items()})
    train, valid = train.iloc[np.flatnonzero(keep)], valid.iloc[np.flatnonzero(keepv)]
    t, y = np.asarray(treatment)[keep], np.asarray(outcome)[keep]
    tv, yv = np.asarray(valid_treatment)[keepv], np.asarray(valid_outcome)[keepv]
    if len(train) < 2 or not len(valid) or len(np.unique(t)) < 2:
        raise numerical.NotEstimable("insufficient_dina_overlap_rows")
    result = fit_effect(
        train=train,
        valid=valid,
        definitions=definitions,
        modifier_ids=[numerical._key(f) for f in definitions],
        treatment=t,
        outcome=y,
        policy=policy,
        seed=seed,
        nuisance=(nt, nv),
        ridge=policy.modifier_ridge_alpha if candidate else None,
    )
    constant = dina.constant_effect(y, t, nt["a"], nt["nu"])
    reduced = float(dina.loss(yv, tv, nv["a"], nv["nu"], constant).mean())
    full = float(dina.loss(yv, tv, nv["a"], nv["nu"], result["delta"]).mean())
    model, design = result["model"], result["design"]
    output = {
        **result["audit"],
        "status": "ok",
        "fit_rows": len(train),
        "heldout_rows": len(valid),
        "tested_interaction_columns": list(design.column_names),
        "encoded_candidate_columns": design.train.shape[1],
        "missingness_interactions": True,
        "feature_group_l2_norms": {
            key: float(
                np.linalg.norm(model.coefficients[np.asarray(design.column_feature_ids) == key])
            )
            for key in set(design.column_feature_ids)
        },
        "heldout_reduced_r_loss": reduced,
        "heldout_full_r_loss": full,
        "heldout_r_loss_improvement": reduced - full,
        "heldout_relative_r_loss_improvement": (reduced - full) / max(reduced, 1e-15),
    }
    if candidate:
        output.update(
            feature_id=numerical._key(definitions[0]),
            name=definitions[0]["name"],
            categorical_interactions_are_grouped=True,
        )
    return output


def label_binary_report(report):
    """Publish native loss aliases while retaining the older selectors' keys."""
    if isinstance(report, list):
        return [label_binary_report(value) for value in report]
    if not isinstance(report, dict):
        return report
    result = {
        key: value if key == "policy" else label_binary_report(value)
        for key, value in report.items()
    }
    native = {
        key.replace("r_loss", "dina_loss"): value
        for key, value in result.items()
        if "r_loss" in key and (key.startswith("heldout_") or key.startswith("mean_"))
    }
    if native:
        result.update(native)
        result.update(
            effect_objective="bernoulli_dina",
            effect_scale="conditional_log_odds_ratio",
            legacy_r_loss_field_semantics="Compatibility aliases of DINA likelihood losses; use dina_loss fields",
        )
    if "effect_modifier_screen" in result:
        result["effect_modifier_screen"].update(
            objective="candidate-wise held-out Bernoulli DINA likelihood gain over a training-fitted constant log odds ratio",
            model_family="candidate_group_dina",
            nuisance_augmentation="all-candidate grouped propensity and arm-outcome nuisances cross-fitted inside training",
            r_learner_comparison="constant versus candidate-varying log odds ratio with the same frozen DINA offset",
            categorical_interaction_handling="all levels and missingness retained together as one feature group",
            missingness_interactions=True,
        )
        result["multivariable_modifier_elastic_net_screen"].update(
            objective="joint grouped elastic-net Bernoulli DINA likelihood",
            model_family="group_elastic_net_dina",
            missingness_interactions=True,
        )
        result["effect_objective"] = "bernoulli_dina"
        result["effect_scale"] = "conditional_log_odds_ratio"
    return result
