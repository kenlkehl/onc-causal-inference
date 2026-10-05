"""Training-only, bounded ontology search for adjustment and heterogeneity.

Original measurements stay available. One label-blind LLM proposal round per
shortlisted variable is followed by paired inner-validation comparisons. The
search selects supplemental definitions, not causal roles or an adjustment set.
Its CV scores are tuning evidence conditional on the upstream discovered catalog;
the outer test fold remains the evaluation of the complete procedure.
"""

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import logging
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from . import stage2_clinical_prompts as prompts
from .stage2_elastic_net_selection import _encode_design
from .stage2_estimand_ontology_config import EstimandOntologyConfig
from .stage2_multi_model_selection import _checkpoint, _frame_hash
from .stage2_role_adjudication import _fingerprint, _write_json

LOGGER = logging.getLogger(__name__)
SCHEMA_VERSION = "stage2_estimand_ontology_v2_dina"
PROMPT_VERSION = "estimand_ontology_proposals_v1_20260926"
DEFINITION_FIELDS = ("description", "value_type", "categories_or_unit",
                     "measurement_definition", "missing_value_rule")
SYSTEM_PROMPT = """Suggest a small set of alternative measurement definitions for one clinical variable in a study of treatment effects.

Purpose

We extract pretreatment measurements from medical records. A variable may help account for differences between patients who received different treatments, or help identify patients whose treatment benefit differs. How a variable is measured can affect both tasks. Your job is to propose clinically meaningful ways to measure the supplied variable. Python will extract each proposal from the records and compare its performance in training and validation splits.

What you receive

The clinical study question, the current definition of one variable, a summary of its extracted values, and a maximum number of alternatives. The study compares the two treatments encoded in that question. Treatment effects are differences in expected outcomes under the two treatments; binary outcomes use a probability difference. Patient outcomes and treatment assignments are unavailable to you.

How to decide

Propose alternatives when the current definition has a concrete ambiguity or loses a clinically useful distinction. An alternative can clarify a pretreatment time window, distinguish a named scale from a generic description, preserve a numerical measurement, or define clinically justified categories. Keep each proposal about the supplied clinical attribute and make it independently extractable from a patient's record. Preserve the study's pretreatment timing. State exactly which documented observation supplies the value and how missing or conflicting observations are handled. Use established category boundaries only when their clinical meaning is clear. Keep missing or unresolved findings as null. A related measurement such as estimated GFR cannot substitute for creatinine clearance. Clinical reasoning must justify every proposal. Return an empty alternatives array when the original definition is clear and no useful alternative is supported.

What to return

Return one JSON object with reason (text) and alternatives (an array). Each alternative contains exactly rationale, description, value_type, categories_or_unit, measurement_definition, and missing_value_rule. value_type is continuous, binary, categorical, or ordinal. categories_or_unit contains one unit for a dimensional continuous variable, an empty array for a unitless variable, exactly two labels for a binary variable, or at least two labels for a categorical or ordinal variable. Put ordinal categories in their clinical order. Python supplies names and identifiers and retains the original measurement."""


class NotEstimable(ValueError):
    """Expected lack of information; preserve original measurements."""


def _key(feature):
    return str(feature["feature_id"])


def proposal_messages(feature, summary, clinical_question, maximum):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Study question\n{clinical_question}\n\n"
            + prompts.feature_text(feature)
            + "\n\nExtracted measurements in the training patients\n"
            + prompts.readable(summary)
            + f"\n\nMaximum alternatives: {maximum}"
        )},
    ]


def validate_proposals(value, *, feature, maximum):
    from .plain_handoff_stage2_analysis import _validate_ontology_refinement

    if (not isinstance(value, dict) or set(value) != {"reason", "alternatives"}
            or not isinstance(value["reason"], str) or not value["reason"].strip()
            or not isinstance(value["alternatives"], list)
            or len(value["alternatives"]) > maximum):
        raise ValueError("return reason and a bounded alternatives array")
    original = {k: feature.get(k) for k in DEFINITION_FIELDS}
    seen = {_fingerprint(original)}
    alternatives = []
    descriptions = set()
    for item in value["alternatives"]:
        if not isinstance(item, dict) or set(item) != {*DEFINITION_FIELDS, "rationale"}:
            raise ValueError("each alternative needs rationale and a complete clinical definition")
        if not isinstance(item["rationale"], str) or not item["rationale"].strip():
            raise ValueError("each alternative needs a clinical rationale")
        for key in ("description", "measurement_definition", "missing_value_rule"):
            if not isinstance(item[key], str) or not item[key].strip():
                raise ValueError(f"alternative {key} must be nonempty text")
        if (not isinstance(item["categories_or_unit"], list)
                or any(not isinstance(c, str) or not c.strip() for c in item["categories_or_unit"])):
            raise ValueError("alternative categories or unit must be nonempty strings")
        decision = _validate_ontology_refinement(
            {"action": "revise", "reason": item["rationale"],
             "definition": {k: item[k] for k in DEFINITION_FIELDS}}, feature=feature)
        definition = {k: decision[k] for k in DEFINITION_FIELDS}
        digest = _fingerprint(definition)
        if digest in seen:
            raise ValueError("alternatives must differ from the original and from each other")
        if definition["description"].casefold() in descriptions:
            raise ValueError("describe each alternative distinctly in clinical language")
        seen.add(digest)
        descriptions.add(definition["description"].casefold())
        alternatives.append({**definition, "rationale": item["rationale"].strip()})
    return {"reason": value["reason"].strip(), "alternatives": alternatives}


def _variant(feature, alternative):
    from .plain_handoff_stage2_analysis import _refresh_conflict_resolution

    # Extraction-specific state must not survive a changed measurement rule.
    result = {k: deepcopy(v) for k, v in feature.items() if k not in {
        "harmonization_plan", "harmonization_fallback", "modeling_strategy",
        "conflict_resolution", "accepted_representations", "estimand_ontology",
    }}
    result.update({k: deepcopy(alternative[k]) for k in DEFINITION_FIELDS})
    suffix = _fingerprint({k: result[k] for k in DEFINITION_FIELDS})[:12]
    result["feature_id"] = f"{_key(feature)}__estimand_{suffix}"
    result["name"] = f"{feature['name']}__estimand_{suffix}"
    result["clinical_label"] = f"{prompts.label(feature)} — {alternative['description']}"
    result["estimand_ontology"] = {
        "schema_version": SCHEMA_VERSION, "source_feature_id": _key(feature),
        "source_feature_name": feature["name"], "rationale": alternative["rationale"],
    }
    return _refresh_conflict_resolution(result)


def _inputs(frame, labels, definitions, splits, *, binary):
    names = [str(f["name"]) for f in definitions]
    keys = [_key(f) for f in definitions]
    if len(set(names)) != len(names) or len(set(keys)) != len(keys):
        raise ValueError("ontology search requires unique feature names and IDs")
    ids = frame._oci_row_id.tolist()
    if (not ids or len(set(ids)) != len(ids)
            or any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) for x in ids)):
        raise ValueError("ontology search requires unique integer training row IDs")
    if list(labels.columns) != ["_oci_row_id", "treatment", "outcome"]:
        raise ValueError("ontology search accepts only training row IDs, treatment, and outcome")
    if labels._oci_row_id.tolist() != ids:
        raise ValueError("ontology labels must match the training matrix row order")
    if not np.isfinite(labels[["treatment", "outcome"]].to_numpy(float)).all():
        raise ValueError("ontology labels must be finite")
    if set(labels.treatment.unique()) != {0, 1}:
        raise ValueError("ontology search requires both binary treatment arms")
    if binary and not set(labels.outcome.unique()) <= {0, 1}:
        raise ValueError("binary outcome must use zero and one")
    counts = Counter()
    for split in splits:
        fit, valid = split["fit_row_ids"], split["heldout_row_ids"]
        if (not fit or not valid or len(set(fit)) != len(fit) or len(set(valid)) != len(valid)
                or set(fit) & set(valid) or set(fit) | set(valid) != set(ids)):
            raise ValueError("ontology inner folds must partition exactly the training rows")
        counts.update(valid)
    if len(splits) < 2 or set(counts) != set(ids) or set(counts.values()) != {1}:
        raise ValueError("ontology validation folds must cover every training patient exactly once")
    return frame[["_oci_row_id", *names]].set_index("_oci_row_id", drop=False), labels.set_index("_oci_row_id")


def _design(train, valid, definitions):
    return _encode_design(train, valid, definitions, categorical_min_count=2)


def _predict(train, valid, target, *, binary, penalty, treatment=None, valid_treatment=None):
    x, xv = np.asarray(train, float), np.asarray(valid, float)
    if treatment is not None:
        x = np.column_stack((x, treatment))
        xv = np.column_stack((xv, valid_treatment))
    if x.shape[1] == 0 or len(np.unique(target)) == 1:
        return np.full(len(xv), float(np.mean(target))), np.zeros(x.shape[1])
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            if binary:
                model = LogisticRegression(C=1.0 / penalty, max_iter=2000, solver="lbfgs")
                model.fit(x, target)
                prediction = model.predict_proba(xv)[:, 1]
            else:
                model = Ridge(alpha=penalty, solver="lsqr", tol=1e-7)
                model.fit(x, target)
                prediction = model.predict(xv)
        except ConvergenceWarning as exc:
            raise NotEstimable("ridge nuisance optimizer did not converge") from exc
    if not np.isfinite(prediction).all():
        raise NotEstimable("nonfinite nuisance predictions")
    return prediction, np.asarray(model.coef_).reshape(-1)


def _loss(y, prediction, binary):
    if not binary:
        return (np.asarray(y) - prediction) ** 2
    p = np.clip(prediction, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def _overlap(e, bounds):
    return (np.asarray(e) >= bounds[0]) & (np.asarray(e) <= bounds[1])


def _r_fit(x, xv, tr, yr, penalty):
    # The constant treatment-effect coefficient is unpenalized. The remaining
    # effect coefficients use ridge shrinkage on training-standardized columns.
    z = np.column_stack((np.ones(len(x)), x)) * tr[:, None]
    if float(tr @ tr) < 1e-10:
        raise NotEstimable("no residual treatment variation")
    # Ridge in the dual is efficient when the reference catalog is very wide.
    denom = float(tr @ tr)
    projected_y = yr - tr * float(tr @ yr) / denom
    residual_x = z[:, 1:] - np.outer(tr, tr @ z[:, 1:] / denom)
    if residual_x.shape[1]:
        model = Ridge(alpha=penalty, fit_intercept=False, solver="lsqr", tol=1e-7)
        model.fit(residual_x, projected_y)
        coef = np.asarray(model.coef_)
    else:
        coef = np.empty(0)
    constant = float(tr @ (yr - z[:, 1:] @ coef)) / denom
    return constant + xv @ coef, coef


def _reference_fold(frame, labels, definitions, split, *, binary, policy, seed, bounds):
    fit, valid = split["fit_row_ids"], split["heldout_row_ids"]
    train, hold = frame.loc[fit], frame.loc[valid]
    t, y = labels.loc[fit, "treatment"].to_numpy(float), labels.loc[fit, "outcome"].to_numpy(float)
    tv, yv = labels.loc[valid, "treatment"].to_numpy(float), labels.loc[valid, "outcome"].to_numpy(float)
    folds = min(policy.nuisance_crossfit_folds, int(min(Counter(t).values())))
    if len(train) < policy.minimum_training_rows or len(np.unique(t)) < 2 or folds < 2:
        raise NotEstimable("insufficient training rows or treatment arms")
    e, m = np.full(len(train), np.nan), np.full(len(train), np.nan)
    arms = {arm: np.full(len(train), np.nan) for arm in (0, 1)}
    arm_valid = {}
    audit = []
    for a, b in StratifiedKFold(folds, shuffle=True, random_state=seed).split(train, t):
        design = _design(train.iloc[a], train.iloc[b], definitions)
        e[b], _ = _predict(design.train, design.valid, t[a], binary=True, penalty=policy.ridge_penalty)
        m[b], _ = _predict(design.train, design.valid, y[a], binary=binary, penalty=policy.ridge_penalty)
        if binary:
            for arm in (0, 1):
                mask = t[a] == arm
                arms[arm][b], _ = _predict(
                    design.train[mask],
                    design.valid,
                    y[a][mask],
                    binary=True,
                    penalty=policy.ridge_penalty,
                )
        audit.append({"fit_row_ids": train.iloc[a]._oci_row_id.tolist(),
                      "validation_row_ids": train.iloc[b]._oci_row_id.tolist()})
    design = _design(train, hold, definitions)
    ev, ce = _predict(design.train, design.valid, t, binary=True, penalty=policy.ridge_penalty)
    mv, _ = _predict(design.train, design.valid, y, binary=binary, penalty=policy.ridge_penalty)
    qv, cq = _predict(design.train, design.valid, y, binary=binary, penalty=policy.ridge_penalty,
                      treatment=t, valid_treatment=tv)
    if binary:
        from ..models import dina

        for arm in (0, 1):
            mask = t == arm
            arm_valid[arm], _ = _predict(
                design.train[mask], design.valid, y[mask], binary=True, penalty=policy.ridge_penalty
            )
        dn = dina.nuisances(e, arms[0], arms[1])
    train_mask, valid_mask = _overlap(e, bounds), _overlap(ev, bounds)
    r_evaluable = (int(train_mask.sum()) >= policy.minimum_overlap_rows
                   and int(valid_mask.sum()) >= policy.minimum_overlap_rows)
    cr = np.zeros(design.train.shape[1])
    if r_evaluable and binary:
        model = dina.fit(
            design.train[train_mask],
            y[train_mask],
            t[train_mask],
            dn["a"][train_mask],
            dn["nu"][train_mask],
            groups=design.column_feature_ids,
            regularization=policy.ridge_penalty / max(1, int(train_mask.sum())),
        )
        cr = model.coefficients
    elif r_evaluable:
        _, cr = _r_fit(design.train[train_mask], design.valid,
                       (t - e)[train_mask], (y - m)[train_mask], policy.ridge_penalty)
    # This prescreen ranks opportunities. It never removes a candidate or role.
    scores = {}
    encoded_by_feature = defaultdict(list)
    for i, key in enumerate(design.column_feature_ids):
        encoded_by_feature[key].append(i)
    for f in definitions:
        indices = encoded_by_feature[_key(f)]
        scores[_key(f)] = {
            "confounder": float(np.linalg.norm(ce[indices]) * np.linalg.norm(cq[indices])),
            "effect_modifier": float(np.linalg.norm(cr[indices])),
        }
    return {
        "fit_row_ids": list(fit),
        "validation_row_ids": list(valid),
        **(
            {
                "training_mu0": arms[0].tolist(),
                "training_mu1": arms[1].tolist(),
                "validation_mu0": arm_valid[0].tolist(),
                "validation_mu1": arm_valid[1].tolist(),
            }
            if binary
            else {}
        ),
        "effect_objective": "bernoulli_dina" if binary else "squared_r_loss",
        "training_propensity": e.tolist(),
        "training_outcome": m.tolist(),
        "validation_propensity": ev.tolist(),
        "validation_outcome": mv.tolist(),
        "propensity_loss": float(_loss(tv, ev, True).mean()),
        "outcome_loss": float(_loss(yv, qv, binary).mean()),
        "overlap_fraction": float(valid_mask.mean()),
        "r_evaluable": bool(r_evaluable),
        "overlap_validation_row_ids": np.asarray(valid)[valid_mask].tolist(),
        "crossfit_audit": audit,
        "screening_scores": scores,
    }


def paired_gain(baseline, candidate, policy):
    """Equal-weight fold comparison; the paired SE is a tuning heuristic."""
    base, alt = np.asarray(baseline, float), np.asarray(candidate, float)
    if len(base) < 2 or base.shape != alt.shape or not np.isfinite([*base, *alt]).all():
        return {"evaluable": False, "accepted": False}
    scale = max(float(base.mean()), 1e-12)
    gains = (base - alt) / scale
    mean = float(gains.mean())
    se = float(gains.std(ddof=1) / np.sqrt(len(gains)))
    wins = float(np.mean(gains > 0))
    return {
        "evaluable": True, "relative_gain": mean, "paired_standard_error": se,
        "winning_fold_fraction": wins,
        "accepted": bool(mean > max(policy.minimum_relative_gain, policy.paired_se_multiplier * se)
                         and wins >= policy.minimum_winning_fold_fraction),
        "formal_error_guarantee": False,
    }


def choose_alternatives(comparisons, policy):
    """Choose at most one alternative per role; original variables remain."""
    eligible = {"confounder": [], "effect_modifier": []}
    decisions = []
    for comparison in comparisons:
        base, alt = comparison["baseline"], comparison["alternative"]
        q = paired_gain([r["outcome_loss"] for r in base], [r["outcome_loss"] for r in alt], policy)
        e = paired_gain([r["propensity_loss"] for r in base], [r["propensity_loss"] for r in alt], policy)
        complete_r = all(r.get("r_loss") is not None for r in [*base, *alt])
        r = paired_gain([x["r_loss"] for x in base], [x["r_loss"] for x in alt], policy) if complete_r else {"evaluable": False, "accepted": False}
        overlap_drop = float(np.mean([b["overlap_fraction"] - a["overlap_fraction"] for b, a in zip(base, alt)]))
        confounder_ok = (q["accepted"] and e["evaluable"]
                         and e["relative_gain"] >= -policy.maximum_propensity_loss_increase
                         and overlap_drop <= policy.maximum_overlap_fraction_drop)
        decision = {"feature_id": comparison["feature_id"], "outcome": q, "propensity": e,
                    "effect": r, "overlap_fraction_drop": overlap_drop,
                    "confounder_eligible": bool(confounder_ok), "modifier_eligible": r["accepted"]}
        decisions.append(decision)
        if confounder_ok:
            eligible["confounder"].append((q["relative_gain"], comparison["feature_id"]))
        if r["accepted"]:
            eligible["effect_modifier"].append((r["relative_gain"], comparison["feature_id"]))
    winners = {role: sorted(rows, key=lambda x: (-x[0], x[1]))[0][1]
               for role, rows in eligible.items() if rows}
    return {"winners": winners, "comparisons": decisions}


def _score_family(frame, labels, definitions, original, alternative, references, *, binary, policy, bounds):
    result = []
    augmented = [*definitions, alternative] if alternative is not None else list(definitions)
    effect_defs = [original, alternative] if alternative is not None else [original]
    for ref in references:
        fit, valid = ref["fit_row_ids"], ref["validation_row_ids"]
        train, hold = frame.loc[fit], frame.loc[valid]
        t, y = labels.loc[fit, "treatment"].to_numpy(float), labels.loc[fit, "outcome"].to_numpy(float)
        tv, yv = labels.loc[valid, "treatment"].to_numpy(float), labels.loc[valid, "outcome"].to_numpy(float)
        if alternative is None:
            qloss, eloss, overlap = ref["outcome_loss"], ref["propensity_loss"], ref["overlap_fraction"]
        else:
            design = _design(train, hold, augmented)
            ep, _ = _predict(design.train, design.valid, t, binary=True, penalty=policy.ridge_penalty)
            qp, _ = _predict(design.train, design.valid, y, binary=binary, penalty=policy.ridge_penalty,
                             treatment=t, valid_treatment=tv)
            qloss, eloss = float(_loss(yv, qp, binary).mean()), float(_loss(tv, ep, True).mean())
            overlap = float(_overlap(ep, bounds).mean())
        rloss = None
        if ref["r_evaluable"]:
            e, m = np.asarray(ref["training_propensity"]), np.asarray(ref["training_outcome"])
            ev, mv = np.asarray(ref["validation_propensity"]), np.asarray(ref["validation_outcome"])
            a, b = _overlap(e, bounds), _overlap(ev, bounds)
            # Fit encoders on precisely the rows used for the effect probe.
            design = _design(train.loc[a], hold.loc[b], effect_defs)
            if binary:
                from ..models import dina
                from .stage2_dina import unpack_nuisances

                dn, dv = unpack_nuisances(ref)
                model = dina.fit(
                    design.train,
                    y[a],
                    t[a],
                    dn["a"][a],
                    dn["nu"][a],
                    groups=design.column_feature_ids,
                    regularization=policy.ridge_penalty / max(1, int(a.sum())),
                )
                rloss = float(
                    dina.loss(
                        yv[b], tv[b], dv["a"][b], dv["nu"][b], model.predict(design.valid)
                    ).mean()
                )
            else:
                tau, _ = _r_fit(
                    design.train, design.valid, (t - e)[a], (y - m)[a], policy.ridge_penalty
                )
                rloss = float(np.mean(((yv - mv)[b] - (tv - ev)[b] * tau) ** 2))
        result.append(
            {
                "outcome_loss": qloss,
                "propensity_loss": eloss,
                "overlap_fraction": overlap,
                "r_loss": rloss,
                "effect_objective": "bernoulli_dina" if binary else "squared_r_loss",
                "validation_row_ids": list(valid),
                "overlap_validation_row_ids": ref["overlap_validation_row_ids"],
            }
        )
    return result


def _restore(frame, definitions, result):
    output = frame.copy()
    for feature in result["accepted_definitions"]:
        output[feature["name"]] = result["accepted_values"][feature["name"]]
    return output.reset_index(drop=True), [*definitions, *result["accepted_definitions"]], result["report"]


def refine_estimand_ontologies(*, extracted_fit, labels, definitions, inner_splits,
                              outcome_type, clinical_question, request_json,
                              extract_alternatives, output_dir, policy=None,
                              source_text_fingerprint, request_identity, seed,
                              propensity_bounds=(0.1, 0.9), proposal_workers=4):
    """The extraction callback receives definitions and a checkpoint directory.

    The caller restricts that callback to outer-training text and applies the
    usual value harmonization before returning (measurements, definitions).
    No dataset or outer-test API is exposed to this numerical component.
    """
    from .plain_handoff_stage2_analysis import feature_summaries, Stage2ResponseValidationError

    policy = policy or EstimandOntologyConfig()
    policy.validate()
    if outcome_type not in {"binary", "continuous"}:
        raise ValueError("ontology search supports binary or continuous outcomes")
    if (len(propensity_bounds) != 2 or not 0 <= propensity_bounds[0] < propensity_bounds[1] <= 1):
        raise ValueError("ontology propensity bounds must satisfy 0 <= lower < upper <= 1")
    # A frozen preselection snapshot may already contain variants from this
    # component. Reconstruct its original catalog before resuming or retuning.
    originals = [deepcopy(f) for f in definitions if not f.get("estimand_ontology")]
    base = extracted_fit[["_oci_row_id", *[f["name"] for f in originals]]].copy()
    if not policy.enabled:
        return base, originals, {"schema_version": SCHEMA_VERSION, "status": "disabled"}
    binary = outcome_type == "binary"
    frame, targets = _inputs(base, labels, originals, inner_splits, binary=binary)
    identity = {
        "schema_version": SCHEMA_VERSION, "prompt_version": PROMPT_VERSION,
        "prompt_sha256": _fingerprint(SYSTEM_PROMPT), "policy": policy.public_dict(),
        "definitions": originals, "measurements": _frame_hash(base),
        "labels": _frame_hash(labels), "inner_splits": list(inner_splits),
        "source_text_fingerprint": source_text_fingerprint,
        "request_identity": request_identity, "clinical_question": clinical_question,
        "outcome_type": outcome_type, "seed": int(seed), "propensity_bounds": list(propensity_bounds),
    }
    digest = _fingerprint(identity)
    root = Path(output_dir)
    directory = root / "runs" / digest
    _write_json(directory / "input.json", {**identity, "input_fingerprint": digest})

    def compute():
        report = {
            "schema_version": SCHEMA_VERSION,
            "input_fingerprint": digest,
            "status": "complete",
            "policy": policy.public_dict(),
            "original_features": len(originals),
            "original_measurements_retained": True,
            "proposal_rounds": 1,
            "outer_test_used": False,
            "oracle_used": False,
            "cv_scope": "tuning_conditional_on_discovered_catalog_and_shortlist",
            "effect_probe": "binary_dina_or_continuous_ridge_r_learner_with_fixed_cross_fitted_reference_nuisances",
            "confounder_probe": "ridge_outcome_given_treatment_and_all_original_candidates",
            "causal_roles_assigned": False,
            "feature_reviews": [],
            "accepted_feature_ids": [],
            "propensity_bounds": list(propensity_bounds),
        }
        try:
            references = []
            for i, split in enumerate(inner_splits):
                LOGGER.info("Estimand ontology: reference nuisance fold %s/%s", i + 1, len(inner_splits))
                references.append(_checkpoint(directory, f"reference_{i + 1:03d}", digest,
                    lambda split=split, i=i: _reference_fold(frame, targets, originals, split,
                        binary=binary, policy=policy, seed=seed + i, bounds=propensity_bounds)))
        except NotEstimable as exc:
            report.update(status="not_estimable", reason=str(exc))
            return {"report": report, "accepted_definitions": [], "accepted_values": {}}
        summaries = {row["feature_id"]: row for row in feature_summaries(base, originals)}
        eligible = [f for f in originals if not f.get("configured_explicit_feature")
                    and summaries[_key(f)]["nonmissing_fraction"] >= policy.minimum_nonmissing_fraction
                    and summaries[_key(f)]["unique_nonmissing"] >= 2]
        score = {role: { _key(f): float(np.mean([r["screening_scores"][_key(f)][role]
                                                for r in references])) for f in eligible}
                 for role in ("confounder", "effect_modifier")}
        shortlist = {role: sorted(values, key=lambda key: (-values[key], key))[:policy.max_features_per_role]
                     for role, values in score.items()}
        report["shortlist"] = shortlist
        report["screening_scores"] = score
        jobs = [f for f in eligible if _key(f) in set(sum(shortlist.values(), []))]
        accepted, accepted_values = [], {}
        existing_names, existing_ids = {f["name"] for f in originals}, {_key(f) for f in originals}
        def prepare(feature):
            feature_dir = directory / "features" / _fingerprint(_key(feature))
            messages = proposal_messages(feature, summaries[_key(feature)], clinical_question,
                                         policy.max_alternatives_per_feature)
            if sum(len(m["content"]) for m in messages) > policy.max_prompt_chars:
                return feature, [], {"feature_id": _key(feature), "status": "prompt_too_large", "winners": {}}

            def propose():
                try:
                    return request_json(messages, lambda value: validate_proposals(value, feature=feature,
                                maximum=policy.max_alternatives_per_feature))
                except Stage2ResponseValidationError as exc:
                    return {"reason": f"Invalid proposal; retained original: {exc}", "alternatives": [],
                            "validation_fallback": True}

            proposal = _checkpoint(feature_dir, "proposal", digest, propose)
            variants = [_variant(feature, a) for a in proposal["alternatives"]]
            if any(v["name"] in existing_names or _key(v) in existing_ids for v in variants):
                raise ValueError("generated ontology identity collides with an existing feature")
            review = {"feature_id": _key(feature), "proposal": proposal, "winners": {}}
            return feature, variants, review

        with ThreadPoolExecutor(max_workers=max(1, min(int(proposal_workers), len(jobs) or 1))) as executor:
            prepared = list(executor.map(prepare, jobs))
        all_variants = [v for _, variants, _ in prepared for v in variants]
        working = frame.copy()
        measurements = pd.DataFrame({"_oci_row_id": base._oci_row_id.tolist()})
        revised_by_id = {}
        if all_variants:
            # Batch all proposed definitions through the existing patient/feature
            # batching machinery; a patient is not revisited per variable family.
            expected_names = {v["name"] for v in all_variants}
            expected_ids = {_key(v) for v in all_variants}
            measurements, revised = extract_alternatives(all_variants, directory / "alternative_measurements")
            if (measurements._oci_row_id.tolist() != base._oci_row_id.tolist()
                    or set(measurements.columns) != {"_oci_row_id", *expected_names}
                    or {_key(v) for v in revised} != expected_ids
                    or len(revised) != len(all_variants)
                    or {v["name"] for v in revised} != expected_names):
                raise ValueError("alternative measurements must cover exactly the training rows and definitions")
            revised_by_id = {_key(v): v for v in revised}
            working = pd.concat([frame, measurements.set_index("_oci_row_id").loc[frame.index]], axis=1)
        for position, (feature, proposed, review) in enumerate(prepared, 1):
            LOGGER.info("Estimand ontology: scoring variable %s/%s %s", position, len(jobs), feature["name"])
            feature_dir = directory / "features" / _fingerprint(_key(feature))
            variants = [revised_by_id[_key(v)] for v in proposed]
            if not variants:
                report["feature_reviews"].append(review)
                continue
            try:
                baseline = _checkpoint(feature_dir, "baseline_scores", digest,
                    lambda: _score_family(working, targets, originals, feature, None, references,
                                         binary=binary, policy=policy, bounds=propensity_bounds))
                comparisons = []
                for variant in variants:
                    coverage = float(working[variant["name"]].notna().mean())
                    if coverage < policy.minimum_nonmissing_fraction or working[variant["name"]].nunique() < 2:
                        review.setdefault("unusable_variants", []).append({"feature_id": _key(variant),
                                                                         "nonmissing_fraction": coverage})
                        continue
                    value_digest = _fingerprint({"input": digest, "definition": variant,
                                                "values": _frame_hash(measurements[["_oci_row_id", variant["name"]]])})
                    alternative = _checkpoint(feature_dir, f"scores_{_fingerprint(_key(variant))}", value_digest,
                        lambda variant=variant: _score_family(working, targets, originals, feature, variant, references,
                                                   binary=binary, policy=policy, bounds=propensity_bounds))
                    comparisons.append({"feature_id": _key(variant), "baseline": baseline, "alternative": alternative})
                review.update(choose_alternatives(comparisons, policy))
            except NotEstimable as exc:
                review.update(status="not_estimable", reason=str(exc))
            for variant in variants:
                supported_roles = [role for role, winner in review["winners"].items() if winner == _key(variant)]
                if not supported_roles:
                    continue
                variant["estimand_ontology"]["supported_uses"] = supported_roles
                variant["estimand_ontology"]["search_fingerprint"] = digest
                accepted.append(variant)
                accepted_values[variant["name"]] = [None if pd.isna(x) else x for x in measurements[variant["name"]].tolist()]
                report["accepted_feature_ids"].append(_key(variant))
            report["feature_reviews"].append(review)
        report["accepted_features"] = len(accepted)
        return {"report": report, "accepted_definitions": accepted, "accepted_values": accepted_values}

    with threadpool_limits(limits=1):
        result = _checkpoint(directory, "result", digest, compute)
    _write_json(root / "report.json", {**result["report"], "checkpoint_directory": str(directory)})
    _write_json(root / "complete.json", {"status": "complete", "input_fingerprint": digest,
                                        "result_sha256": _fingerprint(result)})
    return _restore(base, originals, result)
