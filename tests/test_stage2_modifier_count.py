"""Nested ranking, fixed scoring populations, role protection, and resume integrity."""

from dataclasses import replace
from pathlib import Path

import json

import numpy as np
import pytest

from oci.inference import stage2_modifier_count as count
from oci.inference import stage2_modifier_ranking as ranking
from oci.inference import stage2_multi_model_selection as numerical
from oci.inference.stage2_multi_model_config import (
    MatchedBatchConfig,
    ModifierCountConfig,
    ModifierSelectionConfig,
)
from oci.inference.stage2_role_adjudication import Stage2RoleAdjudicationConfig
from tests.test_stage2_multi_model import sample_inputs


def ranking_request(*args, **kwargs):
    pytest.fail("statistical modifier ranking must not request an LLM")


@pytest.mark.parametrize("binary", [False])
def test_real_nested_selection_scoring_alignment_and_no_refit_resume(tmp_path, monkeypatch, binary):
    arguments = sample_inputs()
    if binary:
        arguments["dataset"]["outcome"] = (arguments["dataset"]["outcome"] > 0).astype(int)
        arguments["outcome_type"] = "binary"
        arguments["extracted_fit"]["other"] = np.where(
            arguments["extracted_fit"]["other"] > 0, "a", "b"
        )
        arguments["definitions"][-1].update(value_type="categorical", categories_or_unit=["a", "b"])
    cfg = ModifierCountConfig(candidate_counts=(0, 1, 2), max_ranked_modifiers=3, forest_seeds=2, concept_review=binary, concept_top_n_per_fold=100)
    arguments["policy"] = replace(
        arguments["policy"],
        min_propensity=0.1,
        max_propensity=0.9,
        multi_model=replace(arguments["policy"].multi_model, repeats=1, modifier_count=cfg,
            modifier_selection=ModifierSelectionConfig(rule="balanced_evidence"),
            matched_batch=MatchedBatchConfig(enabled=True, batch_size=4, bins_per_nuisance=1,
                min_reference_arm=2, max_smd=100, min_delta=0, z_threshold=0,
                train_passes=3, validation_passes=3)),
    )
    numerical_root = tmp_path / "numerical"
    selected, report, _, _ = numerical.select_stage2_features_multi_model(
        **arguments, checkpoint_dir=numerical_root
    )
    role_report = {
        "decisions": report["decisions"]
    }
    nested_calls = []

    def run_nested(values):
        parent = arguments["inner_splits"][len(nested_calls)]
        ids = set(parent["fit_row_ids"])
        assert set(values["extracted_fit"]._oci_row_id) == ids
        assert set(values["dataset"].columns) == {"treatment", "outcome"}
        assert values["dataset"].drop(index=list(ids)).isna().all().all()
        for split in values["inner_splits"]:
            assert set(split["fit_row_ids"]) | set(split["heldout_row_ids"]) == ids
            assert set(split["fit_row_ids"]).isdisjoint(split["heldout_row_ids"])
        nested_calls.append(ids)
        result = numerical.select_stage2_features_multi_model(**values)
        cells = [c for c in result[1]["cells"] if c["family"] == "matched_batch_contrast"]
        assert cells and all(set(c["fit_row_ids"]) | set(c["validation_row_ids"]) <= ids for c in cells)
        return result

    concept_inputs = []
    if cfg.concept_review:
        def concept_review(**kwargs):
            evidence = kwargs["statistical_report"]
            expected_rows = nested_calls[-1] if len(concept_inputs) < 2 else set(range(96))
            assert evidence["cells"]
            for cell in evidence["cells"]:
                assert set(cell["fit_row_ids"]) <= expected_rows
                assert set(cell["validation_row_ids"]) <= expected_rows
            concept_inputs.append(expected_rows)
            assert kwargs["top_n"] == 100
            ranking_args = {k: kwargs[k] for k in ("definitions", "statistical_report", "maximum")}
            ranking_args['output_dir'] = Path(kwargs['output_dir']) / 'representative_ranking'
            return ranking.rank_modifier_candidates(**ranking_args)
        monkeypatch.setattr(count.concepts, "infer_modifier_concepts", concept_review)

    extras = dict(
        selected=selected,
        role_report=role_report,
        statistical_report=report,
        request_json=ranking_request,
        role_policy=Stage2RoleAdjudicationConfig(),
        output_dir=tmp_path / "count",
        numerical_checkpoint_dir=numerical_root,
        model_identity="controlled-response",
        estimation_trees=12,
        run_numerical=run_nested,
    )
    result = count.select_modifier_count(**arguments, **extras)
    if cfg.concept_review:
        assert concept_inputs == [*nested_calls, set(range(96))]
    kept, roles, audit = result
    assert {f["feature_id"]: f["nuisance_model_roles"] for f in kept if "confounder" in f["roles"]} == {
        f["feature_id"]: f["nuisance_model_roles"] for f in selected if "confounder" in f["roles"]
    }
    assert (
        len(nested_calls) == 2
        and audit["boundaries"]["ranking_and_numerical_evidence_nested_within_count_training"]
    )
    assert next(f for f in kept if f["feature_id"] == "c")["roles"][0] == "confounder"
    assert sum("effect_modifier" in f["roles"] for f in kept) == audit["chosen_modifier_count"]
    root = tmp_path / "count" / audit["input_fingerprint"][:20]
    labels = arguments["dataset"]
    for split in arguments["inner_splits"]:
        fold = split["inner_fold"]
        nuisance = json.loads(
            (
                numerical_root
                / report["input_fingerprint"][:20]
                / f"fold_{fold:03d}/nuisances.json"
            ).read_text()
        )["result"]
        valid_ids = split["heldout_row_ids"]
        e, m = (
            np.asarray(nuisance["validation_propensity"]),
            np.asarray(nuisance["validation_outcome"]),
        )
        keep = numerical.linear.propensity_eligibility(e, 0.1, 0.9)
        eligible = np.asarray(valid_ids)[keep].tolist()
        all_ids = []
        for path in (root / f"fold_{fold:03d}").glob("*/size_*/seed_*.json"):
            cell = json.loads(path.read_text())["result"]
            assert cell["validation_row_ids"] == eligible
            expected = (
                labels.loc[eligible, "outcome"].to_numpy()
                - m[keep]
                - (labels.loc[eligible, "treatment"].to_numpy() - e[keep])
                * np.asarray(cell["predictions"])
            ) ** 2
            assert np.allclose(cell["squared_errors"], expected)
            assert np.isclose(cell["r_loss"], expected.mean())
            all_ids.append(cell["validation_row_ids"])
            estimator_name = path.parent.parent.name
            if estimator_name == "linear_interactions":
                # M-01: honest logit(e) and prognostic covariates, unpenalized.
                adjustment = cell["model_audit"]["nuisance_adjustment"]
                assert adjustment["penalized"] is False
                assert adjustment["requested_columns"] == [
                    "logit_propensity", "logit_prognostic_mu0" if binary else "prognostic_mu0",
                ]
                assert set(adjustment["columns"]) <= set(adjustment["requested_columns"])
            elif estimator_name == "causal_forest":
                # M-04: binary-outcome forests always carry the prognostic score.
                assert cell["prognostic_score_in_x"] is binary
                assert cell["constant_design"] is (not binary and cell["encoded_columns"] == 0)
        assert len(all_ids) == (
            16 if binary else 12
        )  # Four sizes: two forest seeds and one interaction fit; common patients.
    options = audit["choice"]["options"]
    chosen_details = options[audit['chosen_estimator']][str(audit['choice']['chosen_additional_count'])]
    assert chosen_details['within_predictive_guardrail']
    eligible_utilities = [o['mean_interaction_utility'] for opts in options.values()
                          for o in opts.values() if o['within_predictive_guardrail']]
    assert chosen_details['mean_interaction_utility'] == pytest.approx(max(eligible_utilities))
    assert "regularization-induced confounding" in audit["methodology_notes"][
        "interaction_model_nuisance_adjustment"]
    assert ("forest_prognostic_score" in audit["methodology_notes"]) is binary
    for fold in audit['folds']:
        assert fold["prognostic_score_source"] in {
            "control_arm_outcome_model_mu0", "marginal_outcome_model_fallback"}
        ranked = json.loads(Path(fold['ranking_path']).read_text())
        utilities = {c['feature_id']: c['interaction_test']['selection_utility']
                     for c in ranked['candidate_scores']}
        keys = [r['feature_id'] for r in fold['ranking']]
        for k, actual in fold['interaction_utilities'].items():
            assert actual == pytest.approx(sum(utilities[key] for key in keys[:int(k)]) / max(len(keys), 1))
    assert set(options) == (
        {"causal_forest", "linear_interactions", "dina"}
        if binary
        else {"causal_forest", "linear_interactions"}
    )
    changed = arguments["dataset"].copy()
    changed.loc[96:, ["treatment", "outcome"]] = 1e12
    changed["oracle"] = "DO_NOT_READ_CHANGED"
    monkeypatch.setattr(count, "_score_prefix", lambda **k: pytest.fail("cached forest refitted"))
    monkeypatch.setattr(
        count, "_score_interactions", lambda **k: pytest.fail("cached interaction model refitted")
    )
    extras.update(
        run_numerical=lambda a: pytest.fail("cached evidence refitted"),
        request_json=lambda *a, **k: pytest.fail("cached LLM called"),
    )
    resumed = count.select_modifier_count(**{**arguments, "dataset": changed}, **extras)
    assert resumed == result
    # Original numerical evidence also survives changes only to the count policy.
    alternate = replace(
        arguments["policy"],
        multi_model=replace(
            arguments["policy"].multi_model,
            modifier_count=replace(cfg, relative_r_loss_tolerance=1e-3),
            interaction_ranking_weight=.25,
        ),
    )
    monkeypatch.setattr(
        numerical,
        "_fit_linear",
        lambda *a, **k: pytest.fail("unchanged evidence refitted"),
    )
    _, reused, _, _ = numerical.select_stage2_features_multi_model(
        **{**arguments, "policy": alternate}, checkpoint_dir=numerical_root
    )
    assert reused["input_fingerprint"] == report["input_fingerprint"]
    assert reused['policy']['multi_model']['interaction_ranking_weight'] == .25
    saved = root / "result.json"
    corrupted = json.loads(saved.read_text())
    corrupted["result"]["report"]["chosen_modifier_count"] += 1
    saved.write_text(json.dumps(corrupted))
    with pytest.raises(ValueError, match="corrupt multi-model checkpoint"):
        count.select_modifier_count(**arguments, **extras)
