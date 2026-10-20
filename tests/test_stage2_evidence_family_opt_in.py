"""The treatment/outcome prediction families are opt-in diagnostics (M-12).

penalized_main and predictive_forest write only treatment- and outcome-role
records. Modifier ranking reads effect-role records, nuisance selection reads
the fold elastic-net supports, and concept-review prompts list effect-role rows
only. Turning the two families on or off must therefore leave every decision
unchanged. The comparisons run the same code twice, so no numerical value is
hard-coded.
"""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

from oci.inference import stage2_modifier_count as count
from oci.inference import stage2_modifier_ranking as ranking
from oci.inference import stage2_multi_model_selection as numerical
from oci.inference.stage2_elastic_net_selection import (
    statistical_selection_config_from_mapping,
)
from oci.inference.stage2_multi_model_config import (
    DIAGNOSTIC_FAMILIES,
    EVIDENCE_FAMILY_POLICY,
    FAMILIES,
    SCHEMA_VERSION,
    MatchedBatchConfig,
    ModifierCountConfig,
    ModifierSelectionConfig,
    Stage2MultiModelConfig,
    multi_model_config_from_mapping,
)
from oci.inference.stage2_role_adjudication import Stage2RoleAdjudicationConfig

BOTH = ("penalized_main", "predictive_forest")
NAMES = (
    "baseline_risk_score", "frailty_index", "biomarker_level", "prior_therapy",
    "noise_alpha", "noise_beta", "noise_gamma", "noise_delta",
)


def _definition(name):
    categorical = name == "prior_therapy"
    return {
        "feature_id": name,
        "name": name,
        "value_type": "categorical" if categorical else "continuous",
        "categories_or_unit": ["yes", "no"] if categorical else ["unitless"],
        "measurement_definition": f"Documented {name.replace('_', ' ')} before treatment.",
        "description": f"Baseline {name.replace('_', ' ')}.",
    }


def _policy(diagnostic_families=(), *, concept_review=False):
    # Built from a JSON-style mapping to exercise list parsing; the reversed
    # order checks canonical normalization.
    multi_model = {
        "repeats": 1,
        "regularization_grid_size": 3,
        "forest_trees": 8,
        "forest_min_samples_leaf": 5,
        "feature_subset_size": 4,
        "permutation_repeats": 2,
        "stability": {"partition_repeats": 1, "subset_partitions": 1},
        # These tests exercise the nested budget search and its concept review.
        "modifier_selection": {"rule": "balanced_evidence"},
        "modifier_count": {
            "candidate_counts": [0, 2, 4],
            "max_ranked_modifiers": 4,
            "forest_seeds": 1,
            "concept_review": concept_review,
        },
        "diagnostic_families": list(reversed(diagnostic_families)),
    }
    return statistical_selection_config_from_mapping({
        "internal_cv_folds": 2,
        "max_iter": 1500,
        "minimum_log10_alpha": -1.5,
        "maximum_log10_alpha": 0,
        "optimization_tolerance": 1e-4,
        "categorical_min_count": 2,
        "min_propensity": 0.05,
        "max_propensity": 0.95,
        "multi_model": multi_model,
    })


def _inputs(policy):
    rng = np.random.default_rng(20261014)
    n = 240
    x = rng.normal(size=(n, 7))
    prior = rng.binomial(1, 0.4, size=n)
    risk, frailty, biomarker = x[:, 0], x[:, 1], x[:, 2]
    t = rng.binomial(1, 1 / (1 + np.exp(-(0.7 * risk - 0.5 * frailty))))
    logit = -0.2 + 0.8 * risk + 0.6 * frailty + t * (0.2 + 1.6 * biomarker + 0.8 * prior)
    y = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    dataset = pd.DataFrame({"treatment": t, "outcome": y, "oracle": ["DO_NOT_READ"] * n})
    extracted = pd.DataFrame(
        {
            "baseline_risk_score": risk,
            "frailty_index": frailty,
            "biomarker_level": biomarker,
            "prior_therapy": np.where(prior == 1, "yes", "no"),
            **{name: x[:, 3 + i] for i, name in enumerate(NAMES[4:])},
        }
    )
    extracted.insert(0, "_oci_row_id", np.arange(n))
    splits = [
        {"inner_fold": i, "fit_row_ids": fit.tolist(), "heldout_row_ids": hold.tolist()}
        for i, (fit, hold) in enumerate(
            KFold(2, shuffle=True, random_state=4).split(extracted), 1
        )
    ]
    return dict(
        dataset=dataset,
        extracted_fit=extracted,
        definitions=[_definition(name) for name in NAMES],
        inner_splits=splits,
        treatment_column="treatment",
        outcome_column="outcome",
        outcome_type="binary",
        seed=31,
        policy=policy,
    )


def _without_diagnostics(evidence):
    return {
        key: [row for row in rows if row["family"] not in DIAGNOSTIC_FAMILIES]
        for key, rows in evidence.items()
    }


def _ranking_view(result):
    """Everything a downstream decision reads; drop identities and audit copies."""
    scores = [
        {
            **{k: v for k, v in c.items() if k != "diagnostic_evidence"},
            "diagnostic_evidence": [
                row for row in c["diagnostic_evidence"] if row["family"] not in DIAGNOSTIC_FAMILIES
            ],
        }
        for c in result["candidate_scores"]
    ]
    return {
        "ranking": result["ranking"],
        "candidate_scores": scores,
        **{k: result[k] for k in (
            "reviewed_candidate_ids", "locked_feature_ids", "measurement_representatives",
            "available_representatives", "criterion",
        )},
    }


def _count_view(kept, roles, audit):
    folds = [
        {k: fold[k] for k in (
            "inner_fold", "ranking_training_row_ids", "scoring_row_ids", "eligible_scoring_row_ids",
            "ranking", "actual_additional_counts", "interaction_utilities", "nuisance_sha256",
        )}
        for fold in audit["folds"]
    ]
    return {
        "kept": kept,
        "decisions": roles["decisions"],
        "choice": audit["choice"],
        "chosen_estimator": audit["chosen_estimator"],
        "chosen_modifier_count": audit["chosen_modifier_count"],
        "requested_additional_budget": audit["requested_additional_budget"],
        "actual_additional_count": audit["actual_additional_count"],
        "full_training_ranking": _ranking_view(audit["full_training_ranking"]),
        "folds": folds,
    }


def _concept_stub(prompts):
    labels = {name.replace("_", " "): name for name in NAMES}

    def request_json(messages, validate, request_kind):
        assert request_kind == "interpretation"
        prompts.append(messages)
        present = [part.split("\n", 1)[0] for part in messages[-1]["content"].split("\n\n")]
        members = [label for label in present if label in labels]
        return {"concepts": [
            {"name": label, "members": [label], "representatives": [label],
             "modifier_recommendation": "retain", "rationale": "Separate clinical concept.",
             "unresolved_questions": ""}
            for label in members
        ]}

    return request_json


def _no_llm(*args, **kwargs):
    pytest.fail("statistical modifier ranking must not request an LLM")


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """Numerical evidence, ranking, and count selection with diagnostics off and on."""
    root = tmp_path_factory.mktemp("evidence_family_opt_in")
    results = {}
    for label, families in (("off", ()), ("on", BOTH)):
        arguments = _inputs(_policy(families))
        numerical_root = root / label / "numerical"
        selected, report, _, _ = numerical.select_stage2_features_multi_model(
            **arguments, checkpoint_dir=numerical_root
        )
        ranked = ranking.rank_modifier_candidates(
            definitions=arguments["definitions"], statistical_report=report,
            output_dir=root / label / "ranking", maximum=8,
        )
        nested_families = []

        def run_nested(values, nested_families=nested_families):
            result = numerical.select_stage2_features_multi_model(**values)
            nested_families.append({c["family"] for c in result[1]["cells"]})
            return result

        common = dict(
            selected=selected,
            role_report={"decisions": report["decisions"]},
            statistical_report=report,
            role_policy=Stage2RoleAdjudicationConfig(),
            numerical_checkpoint_dir=numerical_root,
            model_identity="controlled-response",
            estimation_trees=8,
            run_numerical=run_nested,
        )
        kept, roles, audit = count.select_modifier_count(
            **arguments, **common, request_json=_no_llm, output_dir=root / label / "count",
        )
        # Concept review: the same numerical evidence (cached) with clinical review.
        prompts = []
        concept_arguments = {**arguments, "policy": _policy(families, concept_review=True)}
        concept = count.select_modifier_count(
            **concept_arguments, **common, request_json=_concept_stub(prompts),
            output_dir=root / label / "concept_count",
        )
        results[label] = dict(
            arguments=arguments, selected=selected, report=report, ranked=ranked,
            count=(kept, roles, audit), concept=concept, prompts=prompts,
            nested_families=nested_families,
        )
    return results


def test_default_policy_skips_the_prediction_diagnostics(runs):
    off, on = runs["off"], runs["on"]
    assert off["arguments"]["policy"].multi_model.diagnostic_families == ()
    assert on["arguments"]["policy"].multi_model.diagnostic_families == BOTH
    off_families = {c["family"] for c in off["report"]["cells"]}
    on_families = {c["family"] for c in on["report"]["cells"]}
    assert off_families.isdisjoint(DIAGNOSTIC_FAMILIES)
    assert set(DIAGNOSTIC_FAMILIES) <= on_families
    assert on_families - off_families == set(DIAGNOSTIC_FAMILIES)
    assert off["report"]["model_families"] == [
        f for f in on["report"]["model_families"] if f not in DIAGNOSTIC_FAMILIES
    ]
    assert all(off["report"]["evaluable_cells_by_family"][f] > 0 for f in off_families)
    for family in DIAGNOSTIC_FAMILIES:
        assert on["report"]["evaluable_cells_by_family"][family] > 0
        assert all(
            {row["family"] for row in rows} >= {family}
            for rows in on["report"]["multi_model_evidence"].values()
        )
        # Treatment/outcome records only: nothing for a modifier consumer to read.
        assert {
            r["role"] for c in on["report"]["cells"] if c["family"] == family for r in c["records"]
        } <= {"treatment", "outcome"}
    # Every nested count-training pass follows the same policy.
    assert len(off["nested_families"]) == len(on["nested_families"]) >= 4
    assert all(f.isdisjoint(DIAGNOSTIC_FAMILIES) for f in off["nested_families"])
    assert all(set(DIAGNOSTIC_FAMILIES) <= f for f in on["nested_families"])
    assert off["report"]["policy"]["multi_model"]["diagnostic_families"] == []
    assert on["report"]["policy"]["multi_model"]["diagnostic_families"] == list(BOTH)


def test_diagnostics_do_not_change_other_families_or_nuisance_selection(runs):
    off, on = runs["off"], runs["on"]
    assert [c for c in on["report"]["cells"] if c["family"] not in DIAGNOSTIC_FAMILIES] == (
        off["report"]["cells"]
    )
    assert _without_diagnostics(on["report"]["multi_model_evidence"]) == (
        off["report"]["multi_model_evidence"]
    )
    assert on["selected"] == off["selected"]
    for key in ("decisions", "nuisance_selection", "retained_feature_ids",
                "cross_fitted_nuisance_models", "measurement_groups"):
        assert on["report"][key] == off["report"][key], key


def test_diagnostics_do_not_change_the_modifier_ranking(runs):
    off, on = runs["off"], runs["on"]
    assert off["ranked"]["ranking"]
    assert _ranking_view(on["ranked"]) == _ranking_view(off["ranked"])


def test_diagnostics_do_not_change_count_selection(runs):
    off, on = runs["off"], runs["on"]
    off_view, on_view = _count_view(*off["count"]), _count_view(*on["count"])
    assert off_view["full_training_ranking"]["ranking"]
    assert all(fold["ranking"] for fold in off_view["folds"])
    assert on_view == off_view


def test_concept_review_reads_neither_family_and_does_not_enable_them(runs):
    off, on = runs["off"], runs["on"]
    policy = off["arguments"]["policy"]
    reviewed = replace(policy.multi_model, modifier_count=replace(
        policy.multi_model.modifier_count, concept_review=True))
    assert reviewed.active_families("binary") == policy.multi_model.active_families("binary")
    assert set(reviewed.active_families("binary")).isdisjoint(DIAGNOSTIC_FAMILIES)
    # One review per count-training fold plus the full-training review.
    assert len(off["prompts"]) == len(off["arguments"]["inner_splits"]) + 1
    # Byte-identical clinical inputs: the review never sees either family.
    assert on["prompts"] == off["prompts"]
    rendered = "\n".join(m["content"] for messages in off["prompts"] for m in messages)
    for family in ("Penalized main-effect model", "Predictive forest"):
        assert family not in rendered
    assert _count_view(*on["concept"]) == _count_view(*off["concept"])
    assert all(f.isdisjoint(DIAGNOSTIC_FAMILIES) for f in off["nested_families"])
    assert off["concept"][2]["concept_review_enabled"] is True


def test_active_family_policy_and_parsing():
    # The default orthogonal_tests rule fits only the univariable family (its
    # treatment/outcome associations) besides any requested opt-in family.
    orthogonal = Stage2MultiModelConfig()
    assert orthogonal.computes_effect_evidence is False
    assert orthogonal.active_families("binary") == ("univariable",)
    assert replace(orthogonal, matched_batch=MatchedBatchConfig(enabled=True),
                   diagnostic_families=BOTH).active_families("binary") == (
        "univariable", "penalized_main", "predictive_forest", "matched_batch_contrast")
    assert set(EVIDENCE_FAMILY_POLICY["orthogonal_tests_skipped"]) == set(
        EVIDENCE_FAMILY_POLICY["default_active"]) - {"univariable"}
    default = replace(orthogonal, modifier_selection=ModifierSelectionConfig(
        rule="balanced_evidence"))
    assert default.computes_effect_evidence is True
    assert default.diagnostic_families == ()
    for outcome in (None, "binary", "continuous"):
        active = default.active_families(outcome)
        assert set(active).isdisjoint(DIAGNOSTIC_FAMILIES)
        assert list(active) == [f for f in FAMILIES if f in active]
    assert default.active_families("binary") == tuple(EVIDENCE_FAMILY_POLICY["default_active"])
    assert "dina" not in default.active_families("continuous")
    everything = replace(default, diagnostic_families=BOTH,
                         matched_batch=MatchedBatchConfig(enabled=True))
    everything.validate()
    assert everything.active_families("binary") == FAMILIES
    assert replace(default, diagnostic_families=("predictive_forest",)).active_families(
        "binary") == tuple(f for f in FAMILIES if f not in {"penalized_main", "matched_batch_contrast"})

    parsed = multi_model_config_from_mapping({"diagnostic_families": ["predictive_forest", "penalized_main"]})
    assert parsed.diagnostic_families == BOTH
    public = parsed.public_dict()
    assert public["schema_version"] == SCHEMA_VERSION
    assert public["diagnostic_families"] == list(BOTH)
    assert public["evidence_family_policy"] == EVIDENCE_FAMILY_POLICY
    assert set(EVIDENCE_FAMILY_POLICY["opt_in"]) == {*DIAGNOSTIC_FAMILIES, "matched_batch_contrast"}
    assert multi_model_config_from_mapping(public) == parsed
    assert multi_model_config_from_mapping(default.public_dict()) == default
    assert default.public_dict()["diagnostic_families"] == []
    full = statistical_selection_config_from_mapping({"multi_model": public})
    assert statistical_selection_config_from_mapping(full.public_dict()) == full


@pytest.mark.parametrize("raw, message", [
    ({"diagnostic_families": "predictive_forest"}, "must be a list"),
    ({"diagnostic_families": ["causal_forest"]}, "diagnostic_families"),
    ({"diagnostic_families": ["matched_batch_contrast"]}, "diagnostic_families"),
    ({"diagnostic_families": ["predictive_forest", "predictive_forest"]}, "diagnostic_families"),
    ({"diagnostic_families": [1]}, "diagnostic_families"),
    ({"evidence_family_policy": {"default_active": list(FAMILIES)}}, "evidence_family_policy"),
    ({"schema_version": "stage2_multi_model_selection_v7_adjusted_interactions"}, "schema_version"),
])
def test_invalid_family_policy_is_rejected(raw, message):
    with pytest.raises(ValueError, match=message):
        multi_model_config_from_mapping(raw)


def test_direct_construction_requires_canonical_diagnostic_order():
    with pytest.raises(ValueError, match="diagnostic_families"):
        Stage2MultiModelConfig(diagnostic_families=("predictive_forest", "penalized_main")).validate()
    with pytest.raises(ValueError, match="diagnostic_families"):
        Stage2MultiModelConfig(diagnostic_families=["predictive_forest"]).validate()


def test_example_config_parses_without_diagnostics():
    path = Path(__file__).resolve().parents[1] / "example_configs/research_all_evidence_multi_model.json"
    raw = json.loads(path.read_text())["stage2"]["statistical_selection"]
    config = statistical_selection_config_from_mapping(raw)
    assert config.multi_model.diagnostic_families == ()
    assert set(config.multi_model.active_families("binary")).isdisjoint(DIAGNOSTIC_FAMILIES)
    assert ModifierCountConfig().concept_review is False
