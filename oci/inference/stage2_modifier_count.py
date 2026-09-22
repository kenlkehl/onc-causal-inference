"""Nested, training-only R-loss selection of a ranked modifier budget.

The feature catalog/extractions are frozen upstream. Within each count-validation
split, all seven evidence families and the LLM ranking are rebuilt using only its
training rows. Broad scoring nuisances and eligibility stay fixed across budgets.
This validates Stage 2 conditional on the supplied catalog, not upstream discovery.
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
from .stage2_role_adjudication import _fingerprint, _write_json

LOGGER = logging.getLogger(__name__)
SCHEMA_VERSION = "stage2_nested_modifier_count_v1"


def choose_modifier_count(fold_losses, *, rule):
    """Seeds are averaged within folds; the paired SE is a heuristic, not a test."""
    if rule not in {"minimum_r_loss", "one_standard_error"}:
        raise ValueError("unknown modifier count selection rule")
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
    chosen = (
        best
        if rule == "minimum_r_loss"
        else min(k for k in sizes if options[str(k)]["within_one_se"])
    )
    return {
        "best_mean_count": best,
        "chosen_additional_count": chosen,
        "rule": rule,
        "options": options,
        "fold_losses": {str(k): arrays[k].tolist() for k in sizes},
        "formal_error_guarantee": False,
    }


def _score_prefix(
    *, train, valid, definitions, feature_ids, tr, yr, tvr, yvr, seed, trees
):
    from econml.grf import CausalForest
    from .plain_handoff_stage2_analysis import _FeatureEncoder

    chosen = [f for f in definitions if str(f["feature_id"]) in set(feature_ids)]
    encoder = _FeatureEncoder(chosen).fit(train)
    x, xv = encoder.transform(train), encoder.transform(valid)
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
        raise ValueError(
            "modifier count forest produced nonfinite validation predictions"
        )
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
        "forest_parameters": params,
    }


def _apply_budget(definitions, selected, role_report, ordered, count, locked_modifiers):
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
        rank, evidence = rank_by_id.get(key, (None, {}))
        decision["modifier_count_selection"] = {
            "rank": rank,
            "chosen_additional_count": count,
            "investigator_locked": definition.get("configured_explicit_feature")
            is True,
        }
        if definition.get("configured_explicit_feature") is not True:
            decision["rationale"] = (
                decision.get("rationale", "")
                + f" Modifier role set by nested R-loss prefix selection (rank={rank}, additional budget={count}); confounder role preserved."
            )
            if key in kept_modifiers:
                decision["evidence_ids"] = sorted(
                    set(decision.get("evidence_ids", []))
                    | set(evidence.get("evidence_ids", []))
                )
                decision["modifier_ranking_rationale"] = evidence.get("rationale")
        final_decisions.append(decision)
        if roles:
            feature = deepcopy(previous.get(key, definition))
            feature["roles"] = roles
            if definition.get("configured_explicit_feature") is not True:
                feature["nuisance_model_roles"] = (
                    ["treatment", "outcome"] if "confounder" in roles else []
                )
                feature["selection_source"] = (
                    "multi_model_llm_confounders_nested_r_loss_modifiers"
                )
            retained.append(feature)
    assert {f["feature_id"] for f in retained if "confounder" in f["roles"]} == {
        f["feature_id"] for f in selected if "confounder" in f["roles"]
    }
    report = deepcopy(role_report)
    report["pre_modifier_count_decisions"] = report["decisions"]
    report["decisions"] = final_decisions
    report["retained_feature_ids"] = [f["feature_id"] for f in retained]
    return retained, report


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
):
    """Select a modifier count and preserve every upstream retained confounder."""
    policy.validate()
    cfg = policy.multi_model.modifier_count
    if not cfg.enabled or not definitions:
        return (
            selected,
            role_report,
            {"status": "disabled" if not cfg.enabled else "no_candidates"},
        )
    if not role_policy.enabled:
        raise ValueError("modifier count selection requires LLM role adjudication")
    if (
        isinstance(estimation_trees, bool)
        or not isinstance(estimation_trees, int)
        or estimation_trees < 4
    ):
        raise ValueError(
            "modifier count selection requires at least four estimation trees"
        )
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
        raise ValueError(
            "modifier count received statistical evidence from different inputs"
        )
    from . import plain_handoff_stage2_analysis as analysis

    identity = {
        "schema_version": SCHEMA_VERSION,
        "numerical_input_fingerprint": base_fingerprint,
        "count_policy": cfg.public_dict(),
        "role_policy": role_policy.public_dict(),
        "role_decisions": _fingerprint(role_report),
        "selected_definitions": selected,
        "model": model_identity,
        "estimation_trees": estimation_trees,
        "sources": {
            str(Path(p).name): sha256(Path(p).read_bytes()).hexdigest()
            for p in (__file__, ranking.__file__, analysis.__file__)
        },
    }
    fingerprint = _fingerprint(identity)
    directory = Path(output_dir) / fingerprint[:20]
    _write_json(
        directory / "input.json", {**identity, "input_fingerprint": fingerprint}
    )
    base_numerical_dir = Path(numerical_checkpoint_dir) / base_fingerprint[:20]
    locked_modifiers = [
        str(f["feature_id"])
        for f in definitions
        if f.get("configured_explicit_feature") is True
        and "effect_modifier" in f.get("roles", [])
    ]
    available = sum(
        f.get("configured_explicit_feature") is not True for f in definitions
    )
    maximum = min(available, cfg.max_ranked_modifiers)
    counts = sorted({min(k, maximum) for k in cfg.candidate_counts} | {maximum})
    if run_numerical is None:
        run_numerical = lambda arguments: numerical.select_stage2_features_multi_model(
            **arguments
        )

    def get_ranking(report, root):
        return ranking.rank_modifier_candidates(
            definitions=definitions,
            statistical_report=report,
            request_json=request_json,
            output_dir=root,
            role_policy=role_policy,
            maximum=maximum,
            maximum_chars=policy.multi_model.max_prompt_chars,
            model_identity=model_identity,
        )

    def compute():
        fold_records, losses = [], {k: [] for k in counts}
        if maximum:
            for position, split in enumerate(inner_splits, 1):
                number = int(split.get("inner_fold", position))
                train_ids, valid_ids = split["fit_row_ids"], split["heldout_row_ids"]
                train, valid = (
                    frame.loc[train_ids].reset_index(drop=True),
                    frame.loc[valid_ids].reset_index(drop=True),
                )
                t, y = (
                    labels.loc[train_ids, [treatment_column, outcome_column]]
                    .to_numpy(float)
                    .T
                )
                tv, yv = (
                    labels.loc[valid_ids, [treatment_column, outcome_column]]
                    .to_numpy(float)
                    .T
                )
                fold_seed = int(seed) + position * 10_000
                root = directory / f"fold_{number:03d}"
                LOGGER.info(
                    "Stage 2 modifier count fold=%s: training-only evidence and ranking",
                    number,
                )
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
                    raise ValueError(
                        "insufficient training rows for nested modifier ranking"
                    )
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
                keys = [r["feature_id"] for r in ranked["ranking"]]
                assert len(keys) == maximum
                # These are the already-computed, training-only nuisance models of
                # the parent numerical pass, reused with their original seed/hash.
                nuisance = numerical._checkpoint(
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
                )
                e, m, ev, mv = [
                    np.asarray(nuisance[key], float)
                    for key in (
                        "training_propensity",
                        "training_outcome",
                        "validation_propensity",
                        "validation_outcome",
                    )
                ]
                if not (
                    e.shape == m.shape == t.shape and ev.shape == mv.shape == tv.shape
                ):
                    raise ValueError(
                        "modifier count nuisance rows do not match fold rows"
                    )
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
                fold_records.append(
                    {
                        "inner_fold": number,
                        "ranking_training_row_ids": list(train_ids),
                        "scoring_row_ids": list(valid_ids),
                        "eligible_scoring_row_ids": fv._oci_row_id.tolist(),
                        "nested_evidence_fingerprint": nested_report[
                            "input_fingerprint"
                        ],
                        "ranking": ranked["ranking"],
                        "nuisance_sha256": _fingerprint(nuisance),
                        "ranking_path": str(root / "ranking/ranking.json"),
                    }
                )
                with threadpool_limits(limits=1):
                    for count in counts:
                        ids = [*locked_modifiers, *keys[:count]]
                        seed_losses = []
                        for repeat in range(cfg.forest_seeds):
                            forest_seed = fold_seed + 20_000 + repeat * 1_000_000
                            cell = numerical._checkpoint(
                                root,
                                f"size_{count}/seed_{forest_seed}",
                                fingerprint,
                                lambda: _score_prefix(
                                    train=ft,
                                    valid=fv,
                                    definitions=definitions,
                                    feature_ids=ids,
                                    tr=tr,
                                    yr=yr,
                                    tvr=tvr,
                                    yvr=yvr,
                                    seed=forest_seed,
                                    trees=estimation_trees,
                                ),
                            )
                            seed_losses.append(cell["r_loss"])
                        losses[count].append(float(np.mean(seed_losses)))
                        LOGGER.info(
                            "Stage 2 modifier count fold=%s additional=%s R-loss=%s",
                            number,
                            count,
                            losses[count][-1],
                        )
            choice = choose_modifier_count(losses, rule=cfg.selection_rule)
            full_rank = get_ranking(
                statistical_report, directory / "full_training_ranking"
            )
        else:
            choice = {
                "chosen_additional_count": 0,
                "best_mean_count": 0,
                "rule": cfg.selection_rule,
                "options": {},
                "fold_losses": {},
                "status": "no_unlocked_candidates",
            }
            full_rank = {
                "ranking": [],
                "reviewed_candidate_ids": [],
                "locked_feature_ids": [str(f["feature_id"]) for f in definitions],
            }
        kept, updated_roles = _apply_budget(
            definitions,
            selected,
            role_report,
            full_rank["ranking"],
            choice["chosen_additional_count"],
            locked_modifiers,
        )
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "input_fingerprint": fingerprint,
            "choice": choice,
            "chosen_modifier_count": len(locked_modifiers)
            + choice["chosen_additional_count"],
            "locked_modifier_ids": locked_modifiers,
            "full_training_ranking": full_rank,
            "folds": fold_records,
            "count_policy": cfg.public_dict(),
            "scoring": "mean fold R-loss after averaging forest seeds within each fold",
            "validation_estimator": "historical honest causal forest on fixed all-candidate elastic-net residuals",
            "confounder_policy": "preserve every confounder retained by the full-training role adjudication",
            "boundaries": {
                "ranking_and_numerical_evidence_nested_within_count_training": True,
                "nuisances_and_eligible_rows_fixed_across_counts": True,
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
