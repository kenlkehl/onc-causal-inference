"""Modifier selection: orthogonal tests (default) or the nested budget search.

``orthogonal_tests`` tests every candidate with the pre-specified component
tests of ``stage2_modifier_tests`` on the final-estimation nuisance
construction (training rows only), applies Benjamini-Hochberg to the
Cauchy-combined p-values, and hands the final estimation an estimator policy
(by default the ensemble of every final estimator).

``balanced_evidence`` is the nested predictive/interaction selection of
architecture and modifier budget. The feature catalog/extractions are frozen
upstream. Within each count-validation split, numerical evidence and
statistical ranking use only its training rows. Broad scoring nuisances and
eligibility stay fixed across budgets. This validates Stage 2 conditional on
the supplied catalog, not upstream discovery.
"""

from copy import deepcopy
from hashlib import sha256
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from . import stage2_multi_model_selection as numerical
from . import stage2_modifier_ranking as ranking
from . import stage2_modifier_concepts as concepts
from . import stage2_modeling_pool as modeling
from . import stage2_modifier_tests as modifier_tests
from .stage2_prompt_catalog import PROMPT_VERSION
from .stage2_effect_estimators import (
    NUISANCE_ADJUSTMENT_DESCRIPTION,
    PROGNOSTIC_FOREST_DESCRIPTION,
    ensemble_estimators,
    fit_interaction_effect,
    nuisance_adjustment,
    prognostic_score,
)
from .stage2_dina import fit_effect as fit_dina_effect, unpack_nuisances as unpack_dina_nuisances
from .stage2_role_adjudication import _fingerprint, _write_json

LOGGER = logging.getLogger(__name__)
SCHEMA_VERSION = "stage2_nested_modifier_count_v8_nuisance_adjusted"
ORTHOGONAL_SCHEMA_VERSION = "stage2_orthogonal_modifier_selection_v1"
DEFAULT_PROPENSITY_CLIP = 0.02
TESTING_NUISANCE_SOURCE = "outer_training_inner_out_of_fold_all_candidate_recalibrated"
TESTING_NUISANCE_DESCRIPTION = (
    "Testing nuisances are the final-estimation construction restricted to the "
    "outer-training rows: elastic nets on the complete nuisance-candidate catalog, "
    "fitted in the inner cross-fitting folds with the final-estimation seeds and "
    "standardization, then cross-fitted Platt recalibration. Each training row gets "
    "the prediction of the fold model that did not see it; m_hat = e_hat * mu1_hat + "
    "(1 - e_hat) * mu0_hat. Final estimation reproduces the same training values."
)


def choose_modifier_count(fold_losses):
    """Minimum mean validation loss; seeds are averaged within folds.

    The paired SE is a heuristic, not a test.
    """
    sizes = sorted(fold_losses)
    arrays = {k: np.asarray(fold_losses[k], dtype=float) for k in sizes}
    if not sizes or any(
        a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all() for a in arrays.values()
    ):
        raise ValueError(
            "modifier count requires at least two finite validation-fold losses per size"
        )
    if len({a.shape for a in arrays.values()}) != 1:
        raise ValueError("modifier budgets must use identical validation folds")
    best = min(sizes, key=lambda k: (float(arrays[k].mean()), k))
    options = {}
    for k in sizes:
        difference = arrays[k] - arrays[best]
        excess = float(difference.mean())
        se = float(difference.std(ddof=1) / math.sqrt(len(difference)))
        options[str(k)] = {
            "mean_r_loss": float(arrays[k].mean()),
            "mean_excess_vs_best": excess,
            "paired_standard_error": se,
            "within_one_se": bool(excess <= se + 1e-12),
        }
    return {
        "best_mean_count": best,
        "chosen_additional_count": best,
        "options": options,
        "fold_losses": {str(k): arrays[k].tolist() for k in sizes},
        "formal_error_guarantee": False,
    }


def choose_effect_model(losses, *, interaction_utilities=None,
                        relative_r_loss_tolerance=1e-4):
    """Joint architecture/count choice on paired validation folds.

    Balanced evidence selects an interaction-supported budget among pairs within
    a paired R-loss guardrail.
    """
    architecture_order = {"linear_interactions": 0, "dina": 1, "causal_forest": 2}
    order = sorted(losses, key=lambda pair: (pair[1], architecture_order[pair[0]]))
    indexed = choose_modifier_count({i: losses[pair] for i, pair in enumerate(order)})
    best = order[indexed["best_mean_count"]]
    if (isinstance(relative_r_loss_tolerance, bool)
            or not isinstance(relative_r_loss_tolerance, (int, float))
            or not math.isfinite(relative_r_loss_tolerance)
            or not 0 <= relative_r_loss_tolerance <= 1):
        raise ValueError("relative_r_loss_tolerance must be in [0, 1]")
    best_loss = indexed['options'][str(indexed['best_mean_count'])]['mean_r_loss']
    tolerance = max(abs(best_loss) * relative_r_loss_tolerance, 1e-12)
    utilities = {}
    for k in {pair[1] for pair in order}:
        values = np.asarray(interaction_utilities[k] if interaction_utilities is not None
                            else np.zeros(len(losses[best])), dtype=float)
        if values.shape != (len(losses[best]),) or not np.isfinite(values).all():
            raise ValueError("interaction utilities must match the paired validation folds")
        utilities[k] = float(values.mean())
    eligible = []
    for i, pair in enumerate(order):
        option = indexed['options'][str(i)]
        allowed = option['mean_excess_vs_best'] <= option['paired_standard_error'] + tolerance
        option.update(within_predictive_guardrail=bool(allowed),
                      mean_interaction_utility=utilities[pair[1]])
        if allowed:
            eligible.append(pair)
    strongest = max(utilities[k] for _, k in eligible)
    supported = [pair for pair in eligible if utilities[pair[1]] >= strongest - 1e-12]
    chosen_count = min(k for _, k in supported)
    supported = [pair for pair in supported if pair[1] == chosen_count]
    smallest_loss = min(float(np.mean(losses[pair])) for pair in supported)
    # At the chosen budget, predictive fit chooses the architecture; only
    # numerical/practical ties use the deterministic simplicity order.
    chosen = min((pair for pair in supported
                  if float(np.mean(losses[pair])) <= smallest_loss + tolerance),
                 key=lambda pair: architecture_order[pair[0]])
    guardrail = {
        'rule': 'paired_excess_r_loss <= paired_fold_se + relative_tolerance',
        'absolute_tolerance': tolerance,
        'relative_tolerance': relative_r_loss_tolerance,
        'interaction_utilities': {str(k): v for k, v in sorted(utilities.items())},
        'interaction_evidence_supplied': interaction_utilities is not None,
        'utility_source': 'nested_training_adjusted_interaction_support',
        'formal_error_guarantee': False,
    }
    return {
        "best_mean_estimator": best[0],
        "best_mean_count": best[1],
        "chosen_estimator": chosen[0],
        "chosen_additional_count": chosen[1],
        "rule": "balanced_evidence",
        "options": {
            estimator: {
                str(k): indexed["options"][str(i)]
                for i, (e, k) in enumerate(order)
                if e == estimator
            }
            for estimator in sorted({e for e, _ in order})
        },
        "fold_losses": {
            estimator: {
                str(k): indexed["fold_losses"][str(i)]
                for i, (e, k) in enumerate(order)
                if e == estimator
            }
            for estimator in sorted({e for e, _ in order})
        },
        "tie_break": "interaction_support_then_fewer_modifiers_then_r_loss_then_architecture",
        "predictive_guardrail": guardrail,
        "formal_error_guarantee": False,
    }


def _score_interactions(
    *, train, valid, definitions, feature_ids, t, y, tvr, yvr, binary, policy, seed,
    adjustment=None,
):
    result = fit_interaction_effect(
        train=train,
        valid=valid,
        definitions=definitions,
        modifier_ids=feature_ids,
        treatment=t,
        outcome=y,
        binary=binary,
        policy=policy,
        seed=seed,
        adjustment=adjustment,
    )
    errors = (yvr - tvr * result["tau"]) ** 2
    return {
        "feature_ids": list(feature_ids),
        "fit_n": len(train),
        "validation_n": len(valid),
        "validation_row_ids": valid._oci_row_id.astype(int).tolist(),
        "predictions": result["tau"].tolist(),
        "squared_errors": errors.tolist(),
        "r_loss": float(errors.mean()),
        "model_audit": result["audit"],
    }


def _score_prefix(*, train, valid, definitions, feature_ids, tr, yr, tvr, yvr, seed, trees,
                  prognostic=None):
    """Score a forest on the selected prefix (plus the prognostic column, if given).

    ``prognostic`` is ``(train_values, valid_values)`` of the honest prognostic
    score; the final binary-outcome forest also carries it in X (M-04).
    """
    from econml.grf import CausalForest
    from .plain_handoff_stage2_analysis import _FeatureEncoder

    chosen = [f for f in definitions if str(f["feature_id"]) in set(feature_ids)]
    encoder = _FeatureEncoder(chosen).fit(train)
    x, xv = encoder.transform(train), encoder.transform(valid)
    if prognostic is not None:
        x = np.column_stack((x, np.asarray(prognostic[0], float)))
        xv = np.column_stack((xv, np.asarray(prognostic[1], float)))
    constant = float(tr @ yr / (tr @ tr))
    params = {
        "n_estimators": trees,
        "max_depth": None,
        "min_samples_leaf": 10,
        "max_features": "sqrt",
        "honest": True,
        "inference": True,
        "max_samples": 0.45,
        "subforest_size": next(k for k in (4, 3, 2, 1) if trees % k == 0),
        "n_jobs": 1,
        "random_state": seed,
    }
    if x.shape[1] == 0:
        predicted = np.full(len(tvr), constant)
    else:
        predicted = CausalForest(**params).fit(x, tr, yr).predict(xv).reshape(-1)
    if not np.isfinite(predicted).all():
        raise ValueError("modifier count forest produced nonfinite validation predictions")
    errors = (yvr - tvr * predicted) ** 2
    return {
        "feature_ids": list(feature_ids),
        "encoded_columns": x.shape[1],
        "fit_n": len(train),
        "validation_n": len(valid),
        "validation_row_ids": valid._oci_row_id.astype(int).tolist(),
        "predictions": predicted.tolist(),
        "squared_errors": errors.tolist(),
        "r_loss": float(errors.mean()),
        "constant_effect": constant,
        "constant_design": x.shape[1] == 0,
        "prognostic_score_in_x": prognostic is not None,
        "forest_parameters": params,
    }


def _apply_budget(definitions, selected, role_report, ordered, count, locked_modifiers,
                  *, selection_rule="balanced_evidence"):
    kept_modifiers = set(locked_modifiers) | {r["feature_id"] for r in ordered[:count]}
    previous = {str(f["feature_id"]): f for f in selected}
    decisions = {str(r["feature_id"]): deepcopy(r) for r in role_report["decisions"]}
    rank_by_id = {r["feature_id"]: (i, r) for i, r in enumerate(ordered, 1)}
    retained, final_decisions = [], []
    for definition in definitions:
        key = str(definition["feature_id"])
        old_roles = list(previous.get(key, {}).get("roles", []))
        if definition.get("configured_explicit_feature") is True:
            roles = list(definition.get("roles", []))
        else:
            roles = ["confounder"] if "confounder" in old_roles else []
            if key in kept_modifiers:
                roles.append("effect_modifier")
        decision = decisions[key]
        decision["pre_modifier_count_roles"] = decision["roles"]
        decision["roles"] = roles
        nuisance_roles = list(previous.get(key, {}).get(
            "nuisance_model_roles", ["treatment", "outcome"] if "confounder" in roles else []
        ))
        tasks = nuisance_roles + (["effect"] if "effect_modifier" in roles else [])
        routing = {
            "nuisance_model_roles": nuisance_roles,
            "modeling_tasks": tasks,
            "forest_role": "X" if "effect_modifier" in roles else "W" if nuisance_roles else "excluded",
        }
        decision.update(routing, retained=bool(roles))
        rank, evidence = rank_by_id.get(key, (None, {}))
        decision["modifier_count_selection"] = {
            "rank": rank,
            "chosen_additional_count": count,
            "selection_rule": selection_rule,
            "investigator_locked": definition.get("configured_explicit_feature") is True,
        }
        if definition.get("configured_explicit_feature") is not True:
            decision["rationale"] = decision.get("rationale", "") + (
                f" Modifier role set by nested {selection_rule} prefix selection (rank={rank}, additional budget={count}); nuisance task supports preserved."
                if selection_rule == "balanced_evidence"
                else f" Modifier role set by {selection_rule} Benjamini-Hochberg selection (rank={rank}, discoveries={count}); nuisance task supports preserved."
            )
            if key in kept_modifiers:
                decision["evidence_ids"] = sorted(
                    set(decision.get("evidence_ids", [])) | set(evidence.get("evidence_ids", []))
                )
                decision["modifier_ranking_rationale"] = evidence.get("rationale")
        final_decisions.append(decision)
        if roles:
            feature = deepcopy(previous.get(key, definition))
            feature["roles"] = roles
            feature.update(routing)
            if definition.get("configured_explicit_feature") is not True:
                feature["selection_source"] = (
                    "multi_model_predictive_nuisances_nested_modifier_selection"
                    if selection_rule == "balanced_evidence"
                    else "multi_model_predictive_nuisances_orthogonal_modifier_tests"
                )
                feature["selection_authority"] = "multi_model"
                feature["roles_are_model_routing_not_causal_labels"] = True
                decision.update({k: feature[k] for k in (
                    "selection_source", "selection_authority", "roles_are_model_routing_not_causal_labels"
                )})
            retained.append(feature)
    assert {f["feature_id"] for f in retained if "confounder" in f["roles"]} == {
        f["feature_id"] for f in selected if "confounder" in f["roles"]
    }
    report = deepcopy(role_report)
    report["pre_modifier_count_decisions"] = report["decisions"]
    report["decisions"] = final_decisions
    report["retained_feature_ids"] = [f["feature_id"] for f in retained]
    return retained, report


def _select_by_orthogonal_tests(
    *,
    dataset,
    extracted_fit,
    frame,
    definitions,
    inner_splits,
    treatment_column,
    outcome_column,
    outcome_type,
    seed,
    policy,
    selected,
    role_report,
    base_fingerprint,
    output_dir,
    fit_row_ids,
    propensity_clip,
    tests_path,
    nuisance_model_store=None,
):
    """BH selection on Cauchy-combined orthogonal tests; no nested count search.

    Nuisance and confounder roles come from the one-pass statistical evidence
    (``selected``) and are preserved; only effect-modifier roles change.
    """
    from . import plain_handoff_stage2_analysis as analysis

    chosen = policy.multi_model.modifier_selection
    binary = outcome_type == "binary"
    estimator_policy = chosen.resolved_estimator_policy
    if estimator_policy == "dina" and not binary:
        raise ValueError("modifier_selection.estimator_policy=dina requires a binary outcome")
    weights = (
        {name: float(chosen.weights[name]) for name in modifier_tests.COMPONENTS}
        if chosen.weights is not None
        else modifier_tests.default_weights(outcome_type)
    )
    fit_ids = [
        int(value)
        for value in (fit_row_ids if fit_row_ids is not None else frame["_oci_row_id"])
    ]
    if len(set(fit_ids)) != len(fit_ids) or set(fit_ids) != set(map(int, frame.index)):
        raise ValueError("fit_row_ids must list exactly the outer-training measurement rows")
    locked = {
        str(f["feature_id"]) for f in definitions if f.get("configured_explicit_feature") is True
    }
    locked_modifiers = [
        str(f["feature_id"])
        for f in definitions
        if str(f["feature_id"]) in locked and "effect_modifier" in f.get("roles", [])
    ]
    identity = {
        "schema_version": ORTHOGONAL_SCHEMA_VERSION,
        "tests_schema_version": modifier_tests.SCHEMA_VERSION,
        "rule": "orthogonal_tests",
        "numerical_input_fingerprint": base_fingerprint,
        "modifier_selection": chosen.public_dict(),
        "resolved_weights": weights,
        "resolved_estimator_policy": estimator_policy,
        "propensity_clip": float(propensity_clip),
        "propensity_bounds": {"min": policy.min_propensity, "max": policy.max_propensity},
        "categorical_min_count": policy.categorical_min_count,
        "fit_row_ids": fit_ids,
        "role_decisions": _fingerprint(role_report),
        "selected_definitions": selected,
        "sources": {
            str(Path(p).name): sha256(Path(p).read_bytes()).hexdigest()
            for p in (
                __file__,
                modifier_tests.__file__,
                analysis.__file__,
                Path(__file__).with_name("stage2_statistical_selection.py"),
                Path(__file__).with_name("stage2_interaction_screen.py"),
                Path(__file__).parent.parent / "utils" / "calibration.py",
                Path(__file__).parent.parent / "models" / "elastic_net_nuisance.py",
            )
        },
    }
    fingerprint = _fingerprint(identity)
    directory = Path(output_dir) / fingerprint[:20]
    _write_json(directory / "input.json", {**identity, "input_fingerprint": fingerprint})
    tests_path = Path(tests_path) if tests_path is not None else Path(output_dir) / "modifier_tests.json"
    nuisance_model_store = (
        Path(nuisance_model_store) if nuisance_model_store is not None
        else Path(output_dir) / "nuisance_models"
    )

    def compute():
        # The exact frame the final estimation aligns for its nuisance candidates.
        candidate_fit = analysis._aligned_measurement_frame(
            extracted_fit, fit_ids, definitions, label="training"
        )
        LOGGER.info("Stage 2 modifier tests: training-row final-estimation nuisances")
        nuisances = analysis._outer_training_nuisances(
            dataset=dataset,
            candidate_fit=candidate_fit,
            candidate_definitions=definitions,
            fit_ids=fit_ids,
            inner_splits=inner_splits,
            treatment_column=treatment_column,
            outcome_column=outcome_column,
            binary=binary,
            seed=seed,
            checkpoint_dir=directory / "testing_nuisances",
            checkpoint_fingerprint=fingerprint,
            audit_dir=directory / "testing_nuisance_fit_audits",
            # The final estimation reuses these inner-fold models for held-out rows.
            model_store_root=nuisance_model_store,
            model_store_mode="save",
        )
        e, mu0, mu1 = nuisances["recalibrated"]["fit"]
        t, y = nuisances["treatment"], nuisances["outcome"]
        keep = numerical.linear.propensity_eligibility(
            e, policy.min_propensity, policy.max_propensity
        )
        scores = modifier_tests.testing_scores(
            treatment=t, outcome=y, propensity=e, mu0=mu0, mu1=mu1,
            propensity_clip=propensity_clip,
        )
        nuisance_path = directory / "testing_nuisances.csv"
        analysis._write_frame(nuisance_path, pd.DataFrame({
            "_oci_row_id": fit_ids,
            "inner_fold": nuisances["inner_fold_ids"],
            "propensity": e,
            "mu0": mu0,
            "mu1": mu1,
            "marginal_outcome": scores["marginal"],
            "effect_eligible": keep,
        }))
        LOGGER.info("Stage 2 modifier tests: %s candidates, %s eligible rows",
                    len(definitions), int(keep.sum()))
        records, test_audit = modifier_tests.candidate_tests(
            candidate_fit, definitions, treatment=t, outcome=y, propensity=e, mu0=mu0,
            mu1=mu1, binary=binary, propensity_clip=propensity_clip, eligible=keep,
            min_count=policy.categorical_min_count,
        )
        for record in records:
            record["investigator_locked"] = record["feature_id"] in locked
        records, discovered = modifier_tests.combine_and_select(
            records,
            weights=weights,
            q=chosen.q,
            binary=binary,
            family_ids=[r["feature_id"] for r in records if not r["investigator_locked"]],
            max_modifiers=chosen.max_modifiers,
        )
        for record in records:
            if record["feature_id"] in locked_modifiers:
                record["selected"] = True
                record["selection_reason"] = "investigator_locked_modifier"
            elif record["selected"]:
                record["selection_reason"] = "combined_bh_q_at_most_threshold"
            else:
                record["selection_reason"] = None
            record["nuisance_source"] = TESTING_NUISANCE_SOURCE
        by_id = {r["feature_id"]: r for r in records}
        ordered = [
            {
                "feature_id": key,
                "evidence_ids": [],
                "rationale": (
                    f"Orthogonal modifier tests: combined p={by_id[key]['combined_p_value']:.3g}, "
                    f"BH q={by_id[key]['combined_q_value']:.3g}; labels "
                    f"{', '.join(by_id[key]['labels'])}."
                ),
            }
            for key in discovered
        ]
        kept, updated_roles = _apply_budget(
            definitions, selected, role_report, ordered, len(ordered), locked_modifiers,
            selection_rule="orthogonal_tests",
        )
        for feature in kept:
            if "effect_modifier" in feature["roles"]:
                feature["modifier_evidence"] = modifier_tests.compact_evidence(
                    by_id[str(feature["feature_id"])], rule="orthogonal_tests"
                )
        selected_ids = [*locked_modifiers, *discovered]
        label_counts = {}
        for key in discovered:
            for label in by_id[key]["labels"]:
                label_counts[label] = label_counts.get(label, 0) + 1
        summary = {
            "candidates": len(records),
            "tested_family": sum(r["bh_family"] for r in records),
            "combined_p_available": sum(
                r["combined_p_value"] is not None for r in records if r["bh_family"]
            ),
            "component_p_available": {
                name: sum(
                    r["components"][name].get("p_value") is not None
                    for r in records if r["bh_family"]
                )
                for name in modifier_tests.COMPONENTS
            },
            "discoveries": len(discovered),
            "excluded_by_cap": sum(r["excluded_by_cap"] for r in records),
            "locked_modifiers": len(locked_modifiers),
            "selected_modifiers": len(selected_ids),
            "labels": label_counts,
            "extreme_propensity_signal": sum(
                r["extreme_propensity_signal"] for r in records if r["feature_id"] in discovered
            ),
        }
        testing_nuisances = {
            "source": TESTING_NUISANCE_SOURCE,
            "description": TESTING_NUISANCE_DESCRIPTION,
            "rows": len(fit_ids),
            "eligible_rows": int(keep.sum()),
            "propensity_bounds": {"min": policy.min_propensity, "max": policy.max_propensity},
            "seed_offset": analysis.FINAL_NUISANCE_SEED_OFFSET,
            "predictions_path": str(nuisance_path),
            "model_store": nuisances["model_store"],
            "recalibration": nuisances["recalibrated"]["audit"],
            "fit_audit_summary": analysis._nuisance_fit_audit_summary(
                directory / "testing_nuisance_fit_audits"
            ),
        }
        tests_report = modifier_tests.json_ready({
            "schema_version": modifier_tests.SCHEMA_VERSION,
            "rule": "orthogonal_tests",
            "input_fingerprint": fingerprint,
            "outcome_type": outcome_type,
            "q_threshold": chosen.q,
            "weights": weights,
            "weights_source": "configured" if chosen.weights is not None else "outcome_type_default",
            "max_modifiers": chosen.max_modifiers,
            "propensity_clip": float(propensity_clip),
            "propensity_bounds": {"min": policy.min_propensity, "max": policy.max_propensity},
            "sparse_indicator_min_count": policy.categorical_min_count,
            "nuisance_source": TESTING_NUISANCE_SOURCE,
            "testing_nuisances": testing_nuisances,
            "component_descriptions": modifier_tests.COMPONENT_DESCRIPTIONS,
            "interpretation": modifier_tests.INTERPRETATIONS,
            "test_audit": test_audit,
            "summary": summary,
            "selected_feature_ids": selected_ids,
            "discovered_feature_ids": discovered,
            "locked_modifier_ids": locked_modifiers,
            "candidates": records,
        })
        report = {
            "schema_version": ORTHOGONAL_SCHEMA_VERSION,
            "status": "complete",
            "rule": "orthogonal_tests",
            "input_fingerprint": fingerprint,
            "chosen_estimator": estimator_policy,
            "estimator_policy": estimator_policy,
            "final_estimators": (
                list(ensemble_estimators(outcome_type))
                if estimator_policy == "ensemble" else [estimator_policy]
            ),
            "chosen_modifier_count": len(selected_ids),
            "actual_additional_count": len(discovered),
            "selected_modifier_ids": selected_ids,
            "discovered_feature_ids": discovered,
            "locked_modifier_ids": locked_modifiers,
            "modifier_selection_policy": chosen.public_dict(),
            "resolved_weights": weights,
            "modifier_tests_path": str(tests_path),
            "tests_summary": summary,
            "testing_nuisances": {
                k: v for k, v in testing_nuisances.items()
                if k not in {"recalibration", "fit_audit_summary"}
            },
            "methodology_notes": {
                "modifier_tests": modifier_tests.COMPONENT_DESCRIPTIONS,
                "combination": modifier_tests.INTERPRETATIONS["combined_p_value"],
                "multiplicity": modifier_tests.INTERPRETATIONS["combined_q_value"],
                "training_p_value_optimism": modifier_tests.INTERPRETATIONS[
                    "training_p_value_optimism"
                ],
                "testing_nuisances": TESTING_NUISANCE_DESCRIPTION,
            },
            "propensity_bounds": {"min": policy.min_propensity, "max": policy.max_propensity},
            "confounder_policy": "preserve separate statistically selected nuisance task supports and investigator locks",
            "boundaries": {
                "tests_use_outer_training_rows_only": True,
                "nested_count_search_run": False,
                "oracle_or_outer_test_labels_used": False,
                "candidate_list_built_from_the_same_training_rows": True,
                "conditional_on_frozen_upstream_catalog_and_extractions": True,
            },
            "checkpoint_dir": str(directory),
        }
        updated_roles["modifier_count_selection"] = report
        return {
            "selected": kept, "role_report": updated_roles, "report": report,
            "tests": tests_report,
        }

    result = numerical._checkpoint(directory, "result", fingerprint, compute)
    _write_json(Path(output_dir) / "selection.json", result["report"])
    _write_json(tests_path, result["tests"])
    return result["selected"], result["role_report"], result["report"]


def select_modifier_count(
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
    selected,
    role_report,
    statistical_report,
    request_json,
    role_policy,
    output_dir,
    numerical_checkpoint_dir,
    model_identity,
    estimation_trees,
    run_numerical=None,
    fit_row_ids=None,
    propensity_clip=DEFAULT_PROPENSITY_CLIP,
    tests_path=None,
    nuisance_model_store=None,
):
    """Select modifiers (and the final estimator), preserving each nuisance task's support.

    ``fit_row_ids`` is the outer split's training-row order (the order the final
    estimation uses); ``tests_path`` receives ``modifier_tests.json`` and
    ``nuisance_model_store`` the inner-fold testing-nuisance models (reused by
    the final estimation) under ``orthogonal_tests``.
    """
    policy.validate()
    cfg = policy.multi_model.modifier_count
    if not definitions:
        return selected, role_report, {"status": "no_candidates"}
    if (
        isinstance(estimation_trees, bool)
        or not isinstance(estimation_trees, int)
        or estimation_trees < 4
    ):
        raise ValueError("modifier count selection requires at least four estimation trees")
    definitions = [dict(f) for f in definitions]
    frame, labels = numerical._validated_inputs(
        dataset,
        extracted_fit,
        definitions,
        inner_splits,
        treatment_column,
        outcome_column,
    )
    base_identity = numerical.numerical_identity(
        frame=frame,
        labels=labels,
        definitions=definitions,
        inner_splits=inner_splits,
        outcome_type=outcome_type,
        seed=seed,
        policy=policy,
    )
    base_fingerprint = _fingerprint(base_identity)
    if statistical_report.get("input_fingerprint") != base_fingerprint:
        raise ValueError("modifier count received statistical evidence from different inputs")
    from . import plain_handoff_stage2_analysis as analysis

    if policy.multi_model.modifier_selection.rule == "orthogonal_tests":
        return _select_by_orthogonal_tests(
            dataset=dataset,
            extracted_fit=extracted_fit,
            frame=frame,
            definitions=definitions,
            inner_splits=inner_splits,
            treatment_column=treatment_column,
            outcome_column=outcome_column,
            outcome_type=outcome_type,
            seed=seed,
            policy=policy,
            selected=selected,
            role_report=role_report,
            base_fingerprint=base_fingerprint,
            output_dir=output_dir,
            fit_row_ids=fit_row_ids,
            propensity_clip=propensity_clip,
            tests_path=tests_path,
            nuisance_model_store=nuisance_model_store,
        )

    identity = {
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "numerical_input_fingerprint": base_fingerprint,
        "count_policy": cfg.public_dict(),
        "interaction_ranking_weight": policy.multi_model.interaction_ranking_weight,
        "role_policy": role_policy.public_dict(),
        "role_decisions": _fingerprint(role_report),
        "selected_definitions": selected,
        "model": model_identity,
        "estimation_trees": estimation_trees,
        "sources": {
            str(Path(p).name): sha256(Path(p).read_bytes()).hexdigest()
            for p in (
                __file__,
                ranking.__file__,
                concepts.__file__,
                analysis.__file__,
                Path(__file__).with_name("stage2_effect_estimators.py"),
                Path(__file__).with_name("stage2_dina.py"),
                Path(__file__).parent.parent / "models" / "dina.py",
            )
        },
    }
    fingerprint = _fingerprint(identity)
    directory = Path(output_dir) / fingerprint[:20]
    _write_json(directory / "input.json", {**identity, "input_fingerprint": fingerprint})
    base_numerical_dir = Path(numerical_checkpoint_dir) / base_fingerprint[:20]
    locked_modifiers = [
        str(f["feature_id"])
        for f in definitions
        if f.get("configured_explicit_feature") is True and "effect_modifier" in f.get("roles", [])
    ]
    available = sum(f.get("configured_explicit_feature") is not True for f in definitions)
    maximum = min(available, cfg.max_ranked_modifiers)
    counts = sorted({min(k, maximum) for k in cfg.candidate_counts} | {maximum})
    if run_numerical is None:
        run_numerical = lambda arguments: numerical.select_stage2_features_multi_model(**arguments)

    def get_ranking(report, root):
        report = {**report, "study_context": statistical_report.get("study_context", {"outcome_type": outcome_type})}
        if cfg.concept_review:
            return concepts.infer_modifier_concepts(
                definitions=definitions, statistical_report=report, request_json=request_json,
                output_dir=root / "concepts", role_policy=role_policy,
                top_n=cfg.concept_top_n_per_fold, maximum=maximum,
                maximum_chars=policy.multi_model.max_prompt_chars, model_identity=model_identity)
        return ranking.rank_modifier_candidates(
            definitions=definitions,
            statistical_report=report,
            output_dir=root,
            maximum=maximum,
        )

    def compute():
        estimators = tuple(e for e in cfg.estimators if e != "dina" or outcome_type == "binary")
        if not estimators:
            raise ValueError("no final architectures support this outcome type")
        fold_records, losses = [], {(e, k): [] for e in estimators for k in counts}
        for position, split in enumerate(inner_splits, 1):
            number = int(split.get("inner_fold", position))
            train_ids, valid_ids = split["fit_row_ids"], split["heldout_row_ids"]
            train, valid = (
                frame.loc[train_ids].reset_index(drop=True),
                frame.loc[valid_ids].reset_index(drop=True),
            )
            t, y = labels.loc[train_ids, [treatment_column, outcome_column]].to_numpy(float).T
            tv, yv = labels.loc[valid_ids, [treatment_column, outcome_column]].to_numpy(float).T
            fold_seed = int(seed) + position * 10_000
            root = directory / f"fold_{number:03d}"
            LOGGER.info(
                "Stage 2 modifier count fold=%s: training-only evidence and ranking",
                number,
            )
            if maximum:
                # A physically restricted label table prevents the evidence worker
                # from accessing this fold's scoring labels or any outer-test labels.
                nested_dataset = pd.DataFrame(
                    np.nan,
                    index=range(len(dataset)),
                    columns=[treatment_column, outcome_column],
                )
                nested_dataset.loc[train_ids] = labels.loc[train_ids].to_numpy()
                nested_splits = [
                    {
                        "inner_fold": j,
                        "fit_row_ids": np.asarray(train_ids)[fit].tolist(),
                        "heldout_row_ids": np.asarray(train_ids)[hold].tolist(),
                    }
                    for j, (fit, hold) in enumerate(
                        numerical.linear._crossfit_indices(
                            t,
                            requested_folds=policy.internal_cv_folds,
                            seed=fold_seed + 700_000,
                        ),
                        1,
                    )
                ]
                if len(nested_splits) < 2:
                    raise ValueError("insufficient training rows for nested modifier ranking")
                _, nested_report, _, _ = run_numerical(
                    {
                        "dataset": nested_dataset,
                        "extracted_fit": train,
                        "definitions": definitions,
                        "inner_splits": nested_splits,
                        "treatment_column": treatment_column,
                        "outcome_column": outcome_column,
                        "outcome_type": outcome_type,
                        "seed": fold_seed + 700_000,
                        "policy": policy,
                        "checkpoint_dir": Path(numerical_checkpoint_dir)
                        / "modifier_count_training",
                    }
                )
                ranked = get_ranking(nested_report, root / "ranking")
            else:
                nested_report = {"input_fingerprint": None}
                ranked = {"ranking": []}
            keys = [r["feature_id"] for r in ranked["ranking"]]
            interaction_by_id = {
                c['feature_id']: c.get('interaction_test', {}).get('selection_utility', 0.0)
                for c in ranked.get('candidate_scores', [])
            }
            # Both ordering and this utility use only this split's training
            # evidence. The final full-training ranking cannot select its own
            # validation budget. Missing interaction tests contribute zero.
            interaction_utilities = {
                str(k): float(sum(interaction_by_id.get(key, 0.0) for key in keys[:k])
                              / max(len(keys), 1)) for k in counts
            }
            if not cfg.concept_review and len(keys) != min(maximum, ranked.get("available_representatives", maximum)):
                raise RuntimeError("modifier ranking did not cover the requested budget")
            # These are the already-computed, training-only nuisance models of
            # the parent numerical pass, reused with their original seed/hash.
            nuisance = numerical.checkpoint_future(
                base_numerical_dir,
                f"fold_{number:03d}/nuisances",
                base_fingerprint,
                lambda: numerical._nuisances(
                    train,
                    valid,
                    definitions,
                    t,
                    y,
                    binary=outcome_type == "binary",
                    policy=policy,
                    seed=fold_seed,
                ),
            ).result()
            e, m, ev, mv = [
                np.asarray(nuisance[key], float)
                for key in (
                    "training_propensity",
                    "training_outcome",
                    "validation_propensity",
                    "validation_outcome",
                )
            ]
            if not (e.shape == m.shape == t.shape and ev.shape == mv.shape == tv.shape):
                raise ValueError("modifier count nuisance rows do not match fold rows")
            keep = numerical.linear.propensity_eligibility(
                e, policy.min_propensity, policy.max_propensity
            )
            keepv = numerical.linear.propensity_eligibility(
                ev, policy.min_propensity, policy.max_propensity
            )
            tr, yr, tvr, yvr = (
                (t - e)[keep],
                (y - m)[keep],
                (tv - ev)[keepv],
                (yv - mv)[keepv],
            )
            if len(tr) < 4 or len(tvr) < 2 or float(tr @ tr) <= 1e-12:
                raise ValueError(
                    f"modifier count fold {number} has insufficient propensity-eligible rows or treatment variation"
                )
            ft, fv = (
                train.loc[keep].reset_index(drop=True),
                valid.loc[keepv].reset_index(drop=True),
            )
            # The same honest nuisances (OOF for these training rows, training-only
            # fits for the scoring rows) supply the unpenalized interaction-model
            # covariates (M-01) and the forest's prognostic column (M-04).
            binary_outcome = outcome_type == "binary"
            if "training_mu0" in nuisance and "validation_mu0" in nuisance:
                mu0_train = np.asarray(nuisance["training_mu0"], float)
                mu0_valid = np.asarray(nuisance["validation_mu0"], float)
                prognostic_source = "control_arm_outcome_model_mu0"
            else:
                mu0_train, mu0_valid = m, mv
                prognostic_source = "marginal_outcome_model_fallback"
            adjustment = nuisance_adjustment(
                train_propensity=e[keep],
                valid_propensity=ev[keepv],
                train_mu0=mu0_train[keep],
                valid_mu0=mu0_valid[keepv],
                binary=binary_outcome,
            )
            prognostic = (
                (prognostic_score(mu0_train[keep], binary=True),
                 prognostic_score(mu0_valid[keepv], binary=True))
                if binary_outcome
                else None
            )
            fold_records.append(
                {
                    "inner_fold": number,
                    "ranking_training_row_ids": list(train_ids),
                    "scoring_row_ids": list(valid_ids),
                    "eligible_scoring_row_ids": fv._oci_row_id.tolist(),
                    "nested_evidence_fingerprint": nested_report["input_fingerprint"],
                    "ranking": ranked["ranking"],
                    "concept_review": ranked.get("concept_review"),
                    "actual_additional_counts": {str(k): min(k, len(keys)) for k in counts},
                    "interaction_utilities": interaction_utilities,
                    "nuisance_sha256": _fingerprint(nuisance),
                    "prognostic_score_source": prognostic_source,
                    "ranking_path": str(root / ("ranking/concepts/representative_ranking/ranking.json"
                                                if cfg.concept_review else "ranking/ranking.json")),
                }
            )
            dina_nuisance_cache = {}
            scoring_jobs = []
            with threadpool_limits(limits=1):
                for count in counts:
                    ids = [*locked_modifiers, *keys[:count]]
                    for estimator in estimators:
                        repeats = cfg.forest_seeds if estimator == "causal_forest" else 1
                        for repeat in range(repeats):
                            model_seed = fold_seed + 20_000 + repeat * 1_000_000
                            leaf = f"{estimator}/size_{count}/seed_{model_seed}"
                            if (
                                estimator == "dina" and not dina_nuisance_cache
                                and not (root / f"{leaf}.json").is_file()
                            ):
                                # Compute once with the original first-DINA seed;
                                # every budget then receives the same immutable data.
                                try:
                                    dina_nuisance_cache["raw"] = modeling.call(
                                        lambda: numerical._nuisances(
                                            ft, fv, definitions, t[keep], y[keep],
                                            binary=True, policy=policy,
                                            seed=model_seed + 100_000,
                                        )
                                    )
                                except numerical.NotEstimable as exc:
                                    dina_nuisance_cache["error"] = exc

                            def score():
                                if estimator == "dina":
                                    try:
                                        if "error" in dina_nuisance_cache:
                                            raise dina_nuisance_cache["error"]
                                        shared_nuisance = unpack_dina_nuisances(
                                            dina_nuisance_cache["raw"]
                                        )
                                        fitted = fit_dina_effect(
                                            train=ft,
                                            valid=fv,
                                            definitions=definitions,
                                            modifier_ids=ids,
                                            nuisance=shared_nuisance,
                                            treatment=t[keep],
                                            outcome=y[keep],
                                            policy=policy,
                                            seed=model_seed,
                                        )
                                    except numerical.NotEstimable as exc:
                                        return {
                                            "status": "not_estimable",
                                            "reason": str(exc),
                                            **({"optimizer_diagnostics": exc.diagnostics}
                                               if hasattr(exc, "diagnostics") else {}),
                                            "r_loss": None,
                                            "feature_ids": ids,
                                            "validation_row_ids": fv._oci_row_id.astype(
                                                int
                                            ).tolist(),
                                        }
                                    errors = (yvr - tvr * fitted["tau"]) ** 2
                                    return {
                                        "feature_ids": ids,
                                        "r_loss": float(errors.mean()),
                                        "squared_errors": errors.tolist(),
                                        "predictions": fitted["tau"].tolist(),
                                        "log_odds_ratio": fitted["delta"].tolist(),
                                        "model_audit": fitted["audit"],
                                        "validation_row_ids": fv._oci_row_id.astype(int).tolist(),
                                    }
                                if estimator == "linear_interactions":
                                    return _score_interactions(
                                        train=ft,
                                        valid=fv,
                                        definitions=definitions,
                                        feature_ids=ids,
                                        t=t[keep],
                                        y=y[keep],
                                        tvr=tvr,
                                        yvr=yvr,
                                        binary=outcome_type == "binary",
                                        policy=policy,
                                        seed=model_seed,
                                        adjustment=adjustment,
                                    )
                                return _score_prefix(
                                    train=ft,
                                    valid=fv,
                                    definitions=definitions,
                                    feature_ids=ids,
                                    tr=tr,
                                    yr=yr,
                                    tvr=tvr,
                                    yvr=yvr,
                                    seed=model_seed,
                                    trees=estimation_trees,
                                    prognostic=prognostic,
                                )

                            scoring_jobs.append((estimator, count, numerical.checkpoint_future(
                                root, leaf, fingerprint, score,
                            )))
            seed_losses = {(e, k): [] for e in estimators for k in counts}
            for estimator, count, future in scoring_jobs:
                seed_losses[estimator, count].append(future.result()["r_loss"])
            for count in counts:
                for estimator in estimators:
                    values = seed_losses[estimator, count]
                    losses[estimator, count].append(
                        float(np.mean(values)) if all(v is not None for v in values) else None
                    )
                    LOGGER.info(
                        "Stage 2 effect model fold=%s estimator=%s additional=%s R-loss=%s",
                        number, estimator, count, losses[estimator, count][-1],
                    )
        unavailable = {
            f"{e}/{k}": values
            for (e, k), values in losses.items()
            if any(v is None for v in values)
        }
        evaluable = {
            key: values for key, values in losses.items() if all(v is not None for v in values)
        }
        if not evaluable:
            raise ValueError(
                "no architecture/modifier-count combination is estimable on every validation fold"
            )
        choice = choose_effect_model(
            evaluable,
            interaction_utilities={
                k: [fold['interaction_utilities'][str(k)] for fold in fold_records]
                for k in counts
            },
            relative_r_loss_tolerance=cfg.relative_r_loss_tolerance,
        )
        choice["unavailable_models"] = unavailable
        full_rank = (
            get_ranking(statistical_report, directory / "full_training_ranking")
            if maximum
            else {
                "ranking": [],
                "reviewed_candidate_ids": [],
                "locked_feature_ids": [str(f["feature_id"]) for f in definitions],
            }
        )
        kept, updated_roles = _apply_budget(
            definitions,
            selected,
            role_report,
            full_rank["ranking"],
            choice["chosen_additional_count"],
            locked_modifiers,
            selection_rule=cfg.selection_rule,
        )
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "input_fingerprint": fingerprint,
            "choice": choice,
            "chosen_estimator": choice["chosen_estimator"],
            "chosen_modifier_count": len(locked_modifiers)
            + min(choice["chosen_additional_count"], len(full_rank["ranking"])),
            "requested_additional_budget": choice["chosen_additional_count"],
            "actual_additional_count": min(
                choice["chosen_additional_count"], len(full_rank["ranking"])
            ),
            "concept_review_enabled": cfg.concept_review,
            "locked_modifier_ids": locked_modifiers,
            "full_training_ranking": full_rank,
            "folds": fold_records,
            "count_policy": cfg.public_dict(),
            "scoring": f"joint architecture/count rule={cfg.selection_rule}; paired validation R-loss with nested-training interaction evidence for balanced_evidence; forest seeds averaged within fold; all effects scored as probability differences for binary outcomes",
            "validation_estimators": list(estimators),
            "validation_adjustment": "all frozen candidates for scoring nuisances and interaction main effects; selected prefix for effect inputs/interactions",
            "methodology_notes": {
                "interaction_model_nuisance_adjustment": NUISANCE_ADJUSTMENT_DESCRIPTION,
                **(
                    {"forest_prognostic_score": PROGNOSTIC_FOREST_DESCRIPTION}
                    if outcome_type == "binary"
                    else {}
                ),
            },
            "final_refit_adjustment": "retained confounders and modifiers; forest refits DML nuisances (binary outcomes also carry the honest prognostic score in X); interaction model retunes its outcome penalty with unpenalized honest logit(e_hat) and prognostic covariates",
            "propensity_bounds": {"min": policy.min_propensity, "max": policy.max_propensity},
            "confounder_policy": "preserve separate statistically selected nuisance task supports and investigator locks",
            "boundaries": {
                "ranking_and_numerical_evidence_nested_within_count_training": True,
                "concept_review_nested_within_count_training": cfg.concept_review,
                "nuisances_and_eligible_rows_fixed_across_counts": True,
                "nuisances_and_eligible_rows_fixed_across_architectures": True,
                "oracle_or_outer_test_labels_used": False,
                "conditional_on_frozen_upstream_catalog_and_extractions": True,
                "upstream_discovery_extraction_and_consolidation_renested": False,
            },
            "checkpoint_dir": str(directory),
        }
        updated_roles["modifier_count_selection"] = report
        return {"selected": kept, "role_report": updated_roles, "report": report}

    result = numerical._checkpoint(directory, "result", fingerprint, compute)
    _write_json(Path(output_dir) / "selection.json", result["report"])
    return result["selected"], result["role_report"], result["report"]
