"""Scientific boundaries and end-to-end behavior of the multi-model selector."""

from dataclasses import replace

import json
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

from oci.inference.stage2_elastic_net_selection import (
    Stage2ElasticNetSelectionConfig,
    statistical_selection_config_from_mapping,
)
from oci.inference.stage2_multi_model_config import (
    FAMILIES,
    SCHEMA_VERSION as MULTI_MODEL_SCHEMA_VERSION,
    EvidenceStabilityConfig,
    MatchedBatchConfig,
    ModifierSelectionConfig,
    Stage2MultiModelConfig,
)
from oci.inference.stage2_multi_model_selection import (
    aggregate_evidence,
    select_stage2_features_multi_model,
)
from oci.inference.stage2_multi_model_adjudication import build_multi_model_role_evidence
from oci.inference.stage2_role_adjudication import (
    Stage2RoleAdjudicationConfig,
    role_adjudication_config_from_mapping,
)


def feature(key):
    return {
        "feature_id": key,
        "name": key,
        "value_type": "continuous",
        "categories_or_unit": ["unitless"],
        "measurement_definition": "A measured baseline attribute.",
        "description": "A baseline measurement.",
        "ground_truth": "DO_NOT_LEAK",
        "source_dataset_path": "DO_NOT_LEAK",
    }


def small_policy():
    return Stage2ElasticNetSelectionConfig(
        internal_cv_folds=2,
        max_iter=1500,
        minimum_log10_alpha=-1.5,
        maximum_log10_alpha=0,
        optimization_tolerance=1e-4,
        categorical_min_count=2,
        multi_model=Stage2MultiModelConfig(
            stability=EvidenceStabilityConfig(partition_repeats=1, subset_partitions=1),
            repeats=2,
            regularization_grid_size=3,
            forest_trees=12,
            forest_min_samples_leaf=2,
            feature_subset_size=2,
            permutation_repeats=1,
        ),
    )


def sample_inputs():
    rng = np.random.default_rng(812)
    n = 120
    x = rng.normal(size=(n, 4))
    t = rng.binomial(1, 1 / (1 + np.exp(-x[:, 0])))
    y = x[:, 0] + t * (1.0 + 2.0 * (x[:, 1] > 0)) + rng.normal(scale=0.4, size=n)
    dataset = pd.DataFrame({"treatment": t, "outcome": y, "oracle": ["DO_NOT_READ"] * n})
    extracted = pd.DataFrame(x[:96], columns=["c", "m", "noise", "other"])
    extracted.insert(0, "_oci_row_id", np.arange(96))
    definitions = [feature(k) for k in ["c", "m", "noise", "other"]]
    splits = [
        {"inner_fold": i, "fit_row_ids": tr.tolist(), "heldout_row_ids": va.tolist()}
        for i, (tr, va) in enumerate(KFold(2, shuffle=True, random_state=9).split(extracted), 1)
    ]
    return dict(
        dataset=dataset,
        extracted_fit=extracted,
        definitions=definitions,
        inner_splits=splits,
        treatment_column="treatment",
        outcome_column="outcome",
        outcome_type="continuous",
        seed=12,
        policy=small_policy(),
    )


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("matched", [False, True])
def test_real_models_are_honest_cover_candidates_and_resume_without_refitting(
    tmp_path, monkeypatch, binary, matched
):
    arguments = sample_inputs()
    # Exercise every family, including the opt-in prediction diagnostics; the
    # effect families run under the balanced_evidence rule.
    arguments["policy"] = replace(arguments["policy"], multi_model=replace(
        arguments["policy"].multi_model, diagnostic_families=("penalized_main", "predictive_forest"),
        modifier_selection=ModifierSelectionConfig(rule="balanced_evidence")))
    if matched:
        cfg = MatchedBatchConfig(enabled=True, batch_size=4, bins_per_nuisance=1, min_reference_arm=2,
            max_smd=100, min_delta=0, z_threshold=0, train_passes=3, validation_passes=3)
        arguments["policy"] = replace(arguments["policy"], multi_model=replace(arguments["policy"].multi_model, matched_batch=cfg))
    if binary:
        arguments["outcome_type"] = "binary"
        arguments["dataset"]["outcome"] = (
            arguments["dataset"]["outcome"] > arguments["dataset"]["outcome"].median()
        ).astype(int)
        arguments["extracted_fit"]["other"] = np.where(
            arguments["extracted_fit"]["other"] > 0, "a", "b"
        )
        arguments["definitions"][-1].update(value_type="categorical", categories_or_unit=["a", "b"])
    result = select_stage2_features_multi_model(**arguments, checkpoint_dir=tmp_path)
    _, report, dependencies, latents = result
    assert not latents and len(dependencies) == 4
    assert set(report["evaluable_cells_by_family"]) == set(
        arguments["policy"].multi_model.active_families(arguments["outcome_type"])
    )
    if matched:
        assert set(report["model_families"]) == (
            set(FAMILIES) if binary else set(FAMILIES) - {"dina"}
        )
    assert all(v > 0 for v in report["evaluable_cells_by_family"].values())
    if matched:
        legacy_policy = replace(arguments["policy"], multi_model=replace(
            arguments["policy"].multi_model, matched_batch=MatchedBatchConfig()))
        _, legacy, _, _ = select_stage2_features_multi_model(**{**arguments, "policy": legacy_policy})
        assert [c for c in report["cells"] if c["family"] != "matched_batch_contrast"] == legacy["cells"]
    assert {r["_oci_row_id"] for r in report["cross_fitted_nuisance_models"]["predictions"]} == set(
        range(96)
    )
    for cell in report["cells"]:
        assert set(cell["fit_row_ids"]).isdisjoint(cell["validation_row_ids"])
        assert set(cell["fit_row_ids"]) | set(cell["validation_row_ids"]) <= set(range(96))
    for feature_id, summaries in report["multi_model_evidence"].items():
        for family in ("causal_forest", "predictive_forest"):
            rows = [r for r in summaries if r["family"] == family]
            assert rows and all(row["exposures"] == 4 for row in rows)
    # No outside label or source oracle column participates in the identity or fitting.
    changed = arguments["dataset"].copy()
    changed.loc[96:, ["treatment", "outcome"]] = 1e12
    changed["oracle"] = "NEW_POISON"
    monkeypatch.setattr(
        "oci.inference.stage2_multi_model_selection._fit_linear",
        lambda *a, **k: pytest.fail("completed model checkpoint was refitted"),
    )
    monkeypatch.setattr("oci.inference.stage2_matched_batch.score_candidates",
                        lambda *a, **k: pytest.fail("completed batch checkpoint was refitted"))
    resumed = select_stage2_features_multi_model(
        **{**arguments, "dataset": changed}, checkpoint_dir=tmp_path
    )
    assert resumed == result
    nuisance_files = list(tmp_path.glob("*/fold_*/nuisances.json"))
    for path in nuisance_files:
        cached = json.loads(path.read_text())["result"]
        for subfold in cached["crossfit_audit"]:
            assert set(subfold["fit_row_ids"]).isdisjoint(subfold["heldout_row_ids"])
            assert all(m["converged"] for m in subfold["models"])


def test_invalid_inner_boundary_fails_before_fitting(tmp_path):
    arguments = sample_inputs()
    arguments["inner_splits"][0]["heldout_row_ids"].append(100)
    with pytest.raises(ValueError, match="outer-training"):
        select_stage2_features_multi_model(**arguments, checkpoint_dir=tmp_path)
    arguments = sample_inputs()
    arguments["dataset"]["treatment"] = 0
    with pytest.raises(ValueError, match="both binary treatment arms"):
        select_stage2_features_multi_model(**arguments, checkpoint_dir=tmp_path)


def test_all_modifier_evidence_uses_overlap_while_associations_keep_all_rows(monkeypatch):
    from oci.inference import stage2_multi_model_selection as selection

    arguments = sample_inputs()
    arguments["policy"] = replace(
        arguments["policy"],
        min_propensity=0.1,
        max_propensity=0.9,
        multi_model=replace(arguments["policy"].multi_model, repeats=1,
                            modifier_selection=ModifierSelectionConfig(rule="balanced_evidence")),
    )
    arguments["policy"] = replace(arguments["policy"], multi_model=replace(arguments["policy"].multi_model,
        matched_batch=MatchedBatchConfig(enabled=True, batch_size=4, bins_per_nuisance=1,
            min_reference_arm=2, train_passes=2, validation_passes=2)))
    probabilities = np.array([0.05, 0.1, 0.5, 0.9, 0.95, 0.5])

    def p(frame):
        return probabilities[frame._oci_row_id.to_numpy() % len(probabilities)]

    # Hold honest scoring nuisances fixed to isolate the downstream eligibility boundary.
    monkeypatch.setattr(
        selection,
        "_nuisances",
        lambda train, valid, *a, **k: {
            "training_propensity": p(train).tolist(),
            "training_outcome": np.zeros(len(train)).tolist(),
            "validation_propensity": p(valid).tolist(),
            "validation_outcome": np.zeros(len(valid)).tolist(),
            "crossfit_audit": [],
            "validation_fit_audits": [],
                "predictive_support": {"task_feature_ids": {"treatment": [], "outcome": []}},
        },
    )
    _, original, _, _ = select_stage2_features_multi_model(**arguments)
    excluded = ~selection.linear.propensity_eligibility(p(arguments["extracted_fit"]), 0.1, 0.9)
    changed = arguments["dataset"].copy()
    changed.loc[np.flatnonzero(excluded), "outcome"] += 100
    _, poisoned, _, _ = select_stage2_features_multi_model(**{**arguments, "dataset": changed})
    for old, new in zip(original["cells"], poisoned["cells"]):
        if old["family"] == "univariable":
            assert old["audit"]["association_fit_row_ids"] == old["fit_row_ids"]
            effect_ids = old["audit"]["effect_fit_row_ids"]
            assert len(effect_ids) < len(old["fit_row_ids"])
        elif old["family"] in {
            "penalized_interactions",
            "orthogonal_linear",
            "univariable_rlearner",
            "causal_forest",
            "matched_batch_contrast",
        }:
            effect_ids = old["fit_row_ids"]
            assert all(0.1 <= probabilities[i % 6] <= 0.9 for i in old["validation_row_ids"])
        else:
            assert len(old["fit_row_ids"]) == 48
            continue
        assert all(0.1 <= probabilities[i % 6] <= 0.9 for i in effect_ids)
        assert {probabilities[i % 6] for i in effect_ids} >= {0.1, 0.9}
        assert [r for r in old["records"] if r["role"] == "effect"] == [
            r for r in new["records"] if r["role"] == "effect"
        ]
    empty = selection._univariable(
        arguments["extracted_fit"],
        arguments["definitions"],
        arguments["dataset"].treatment.to_numpy(float)[:96],
        arguments["dataset"].outcome.to_numpy(float)[:96],
        binary=False,
        policy=arguments["policy"],
        effect_keep=np.zeros(96, bool),
        nuisance={"e": np.full(96, 0.5), "m": np.zeros(96)},
    )
    assert all(r["status"] == "not_evaluable" for r in empty["records"] if r["role"] == "effect")


def test_missing_fit_is_not_a_negative_stability_vote(monkeypatch):
    cells = [
        {
            "family": "causal_forest",
            "inner_fold": i,
            "records": [
                {
                    "feature_id": "a",
                    "role": "effect",
                    "status": status,
                    "score": value,
                    "selected": selected,
                }
            ],
        }
        for i, status, value, selected in [
            (1, "ok", 0.2, True),
            (2, "not_evaluable", None, None),
            (3, "ok", -0.1, False),
        ]
    ]
    row = aggregate_evidence(cells, ["a", "b"])["a"][0]
    assert (row["exposures"], row["evaluated"], row["not_evaluable"], row["support_fraction"]) == (
        3,
        2,
        1,
        0.5,
    )
    assert row["folds"][1] == {"inner_fold": 2, "evaluated": 0, "supported": 0}
    # An unreliable outcome test must not erase an evaluable treatment test.
    from oci.inference import stage2_multi_model_selection as selection

    calls = []

    def nested_test(*args):
        calls.append(None)
        if len(calls) == 2:
            warnings.warn("Perfect separation detected", UserWarning)
        return 0.01, {"status": "ok"}

    monkeypatch.setattr(selection.linear, "_binary_nested_p_value", nested_test)
    arguments = sample_inputs()
    labels = arguments["dataset"].iloc[:96]
    result = selection._univariable(
        arguments["extracted_fit"],
        arguments["definitions"][:1],
        labels.treatment.to_numpy(dtype=float),
        (labels.outcome > labels.outcome.median()).to_numpy(dtype=float),
        binary=True,
        policy=arguments["policy"],
        nuisance={"e": np.full(96, 0.5), "m": np.full(96, 0.5)},
    )
    treatment = next(r for r in result["records"] if r["role"] == "treatment")
    outcome = next(r for r in result["records"] if r["role"] == "outcome")
    assert treatment["status"] == "ok" and treatment["selected"]
    assert outcome["status"] == "not_evaluable" and outcome["selected"] is None


def role_fixture():
    definitions = [feature(f"f{i}") for i in range(7)]
    definitions[0].update(configured_explicit_feature=True, roles=["confounder"])
    records = []
    for f in definitions:
        for role in ("treatment", "outcome", "effect"):
            records.append(
                {
                    "feature_id": f["feature_id"],
                    "role": role,
                    "status": "ok",
                    "selected": True,
                    "score": 0.2,
                }
            )
    report = {
        "policy": small_policy().public_dict(),
        "multi_model_evidence": aggregate_evidence(
            [{"family": "univariable", "inner_fold": 1, "records": records}],
            [f["feature_id"] for f in definitions],
        ),
        "row_values": "DO_NOT_LEAK",
        "dataset_path": "DO_NOT_LEAK",
        "oracle": "DO_NOT_LEAK",
    }
    return definitions, report


def test_concept_review_evidence_uses_allowlisted_aggregate_inputs_only():
    definitions, report = role_fixture()
    definitions[1].update(oracle_role="DO_NOT_LEAK", data_generation_notes="DO_NOT_LEAK")
    policy = Stage2RoleAdjudicationConfig()
    evidence = build_multi_model_role_evidence(
        definitions=definitions, statistical_report=report, policy=policy
    )
    rendered = json.dumps(evidence)
    assert "DO_NOT_LEAK" not in rendered
    for key in ("ground_truth", "source_dataset_path", "oracle_role", "data_generation_notes", "row_values"):
        assert key not in rendered
    assert evidence["evidence_boundary"]["outer_heldout_rows_are_excluded"] is True
    assert evidence["evidence_boundary"]["oracle_columns_are_excluded"] is True
    assert [c["feature_id"] for c in evidence["candidates"]] == [f["feature_id"] for f in definitions]
    locked = evidence["candidates"][0]["definition"]
    assert locked["investigator_locked"] is True and locked["configured_roles"] == ["confounder"]
    with pytest.raises(ValueError, match="exactly once"):
        build_multi_model_role_evidence(
            definitions=definitions[:-1], statistical_report=report, policy=policy
        )


def test_default_policy_roundtrips_and_is_multi_model_only():
    default = statistical_selection_config_from_mapping({})
    public = default.public_dict()
    assert "selection_mode" not in public
    assert public["multi_model"]["schema_version"] == MULTI_MODEL_SCHEMA_VERSION
    count_policy = default.multi_model.modifier_count
    assert count_policy.estimators == ("causal_forest", "linear_interactions", "dina")
    assert count_policy.selection_rule == "balanced_evidence"
    assert "enabled" not in count_policy.public_dict()
    assert default.multi_model.interaction_ranking_weight == 0.5
    # The former mode selector and count-search flag are accepted only as no-ops.
    assert statistical_selection_config_from_mapping(
        {"selection_mode": "multi_model", "multi_model": {"modifier_count": {"enabled": True}}}
    ) == default
    custom = statistical_selection_config_from_mapping({"multi_model": {"repeats": 2}})
    assert custom.multi_model.repeats == 2
    for config in (default, custom):
        assert statistical_selection_config_from_mapping(config.public_dict()) == config


@pytest.mark.parametrize("name, value", [
    ("nuisance_selection_rule", "any_fold"),
    ("modifier_selection_rule", "any_fold"),
    ("nuisance_selection_frequency", 0.5),
    ("modifier_selection_frequency", 0.5),
    ("regularization_grid_size", 16),
    ("one_standard_error_rule", True),
    ("nuisance_prediction_one_standard_error_rule", True),
    ("univariable_confounder_p_value_threshold", 0.05),
    ("univariable_confounder_q_value_threshold", 0.1),
    ("nuisance_forest_trees", 100),
    ("nuisance_forest_min_samples_leaf", 5),
    ("modifier_top_n_per_inner_fold", 10),
    ("modifier_min_fold_r_loss_improvement", 0.0),
    ("modifier_min_mean_r_loss_improvement", 0.0),
    ("modifier_min_positive_fold_fraction", 0.5),
])
def test_former_statistical_selection_fields_are_rejected(name, value):
    # Saved configurations carrying removed selector fields fail closed.
    with pytest.raises(
        ValueError, match=rf"stage2\.statistical_selection contains unsupported fields: \['{name}'\]"
    ):
        statistical_selection_config_from_mapping({name: value})


@pytest.mark.parametrize("name, value", [("enabled", True), ("max_candidates_per_request", 20)])
def test_former_role_adjudication_fields_are_rejected(name, value):
    with pytest.raises(
        ValueError, match=rf"stage2\.role_adjudication contains unsupported fields: \['{name}'\]"
    ):
        role_adjudication_config_from_mapping({name: value})
    with pytest.raises(ValueError, match="unsupported"):
        role_adjudication_config_from_mapping({"unknown": 1})
    assert role_adjudication_config_from_mapping({}) == Stage2RoleAdjudicationConfig()


@pytest.mark.parametrize(
    "raw, message",
    [
        ({"selection_mode": "llm_roles"}, "only 'multi_model'"),
        ({"selection_mode": "independent_tasks"}, "only 'multi_model'"),
        ({"unknown": True}, "unsupported"),
        ({"multi_model": {"unknown": True}}, "unsupported"),
        (
            {"multi_model": {"schema_version": "stage2_multi_model_selection_v6_predictive_nuisances"}},
            "schema_version",
        ),
        ({"multi_model": {"modifier_count": {"enabled": False}}}, "no longer supported"),
        ({"multi_model": {"modifier_count": {"enabled": "yes"}}}, "no longer supported"),
        ({"multi_model": {"modifier_count": {"selection_rule": "minimum_r_loss"}}}, "selection_rule"),
        ({"multi_model": {"modifier_count": {"selection_rule": "oracle"}}}, "selection_rule"),
        ({"multi_model": {"modifier_count": {"candidate_counts": [4, 0]}}}, "candidate_counts"),
        ({"multi_model": {"modifier_count": {"candidate_counts": [0, True]}}}, "candidate_counts"),
        *[
            ({"multi_model": {"modifier_count": {"estimators": estimators}}}, "estimators")
            for estimators in ([], ["unknown"], ["causal_forest", "causal_forest"], [True])
        ],
    ],
)
def test_invalid_selection_policy_is_rejected(raw, message):
    with pytest.raises(ValueError, match=message):
        statistical_selection_config_from_mapping(raw)


@pytest.mark.parametrize("value", [-0.1, 1.1, True, float("nan")])
def test_invalid_numeric_policy_values_are_rejected(value):
    multi = statistical_selection_config_from_mapping({}).multi_model
    for invalid in (
        replace(multi, interaction_ranking_weight=value),
        replace(multi, row_fraction=value),
        replace(multi, modifier_count=replace(multi.modifier_count, relative_r_loss_tolerance=value)),
    ):
        with pytest.raises(ValueError):
            invalid.validate()


def test_new_example_uses_predictive_nuisances_and_freezes_resume_policy(tmp_path):
    from oci.inference.research_all_evidence_workflow import (
        _stage2_reselection_policy_fingerprint,
        compile_config,
    )
    from oci.inference.stage2_preflight import validate_selection_resume

    path = (
        Path(__file__).resolve().parents[1]
        / "example_configs/research_all_evidence_multi_model.json"
    )
    raw = json.loads(path.read_text())
    config = compile_config(raw, config_dir=path.parent)
    assert config.stage2.statistical_selection.multi_model.modifier_count.selection_rule == 'balanced_evidence'
    assert config.stage2.statistical_selection.multi_model.interaction_ranking_weight == .5
    assert (config.stage2.min_propensity, config.stage2.max_propensity) == (0.1, 0.9)
    root = tmp_path / "stage2"
    root.mkdir()
    saved = {
        "statistical_selection": config.stage2.statistical_selection.public_dict(),
        "min_propensity": config.stage2.min_propensity,
        "max_propensity": config.stage2.max_propensity,
        "estimation_trees": config.stage2.estimation_trees,
    }
    (root / "config.json").write_text(json.dumps(saved))
    report_dir = root / "outer_001/selection"
    report_dir.mkdir(parents=True)
    (report_dir / "elastic_net_selection.json").write_text(
        json.dumps({"selection_authority": "multi_model"})
    )
    validate_selection_resume(SimpleNamespace(stage2=config.stage2, output_dir=tmp_path))
    # Under the example's orthogonal_tests rule the trees affect final estimation only.
    changed_trees = replace(config.stage2, estimation_trees=config.stage2.estimation_trees * 2)
    validate_selection_resume(SimpleNamespace(stage2=changed_trees, output_dir=tmp_path))
    assert _stage2_reselection_policy_fingerprint(config) == _stage2_reselection_policy_fingerprint(
        replace(config, stage2=changed_trees)
    )
    # balanced_evidence scores forests of that size during its count search.
    balanced_selection = replace(
        config.stage2.statistical_selection,
        multi_model=replace(
            config.stage2.statistical_selection.multi_model,
            modifier_selection=ModifierSelectionConfig(rule="balanced_evidence"),
        ),
    )
    balanced = replace(config.stage2, statistical_selection=balanced_selection)
    (root / "config.json").write_text(json.dumps(
        {**saved, "statistical_selection": balanced_selection.public_dict()}))
    validate_selection_resume(SimpleNamespace(stage2=balanced, output_dir=tmp_path))
    balanced_trees = replace(balanced, estimation_trees=balanced.estimation_trees * 2)
    with pytest.raises(RuntimeError, match="modifier-count estimation_trees changed"):
        validate_selection_resume(SimpleNamespace(stage2=balanced_trees, output_dir=tmp_path))
    assert _stage2_reselection_policy_fingerprint(
        replace(config, stage2=balanced)
    ) != _stage2_reselection_policy_fingerprint(replace(config, stage2=balanced_trees))
    (root / "config.json").write_text(json.dumps(saved))
    changed = replace(
        config.stage2,
        statistical_selection=replace(
            config.stage2.statistical_selection,
            multi_model=replace(config.stage2.statistical_selection.multi_model, repeats=7),
        ),
    )
    with pytest.raises(RuntimeError, match="multi-model policy changed"):
        validate_selection_resume(SimpleNamespace(stage2=changed, output_dir=tmp_path))
