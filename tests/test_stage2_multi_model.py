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
    select_stage2_features_elastic_net,
    statistical_selection_config_from_mapping,
)
from oci.inference.stage2_multi_model_config import FAMILIES, Stage2MultiModelConfig
from oci.inference.stage2_multi_model_selection import (
    aggregate_evidence,
    candidate_subsets,
    grouped_permutation_gain,
)
from oci.inference.stage2_multi_model_adjudication import (
    _decision_validator,
    _themes_validator,
    build_multi_model_role_evidence,
)
from oci.inference.stage2_role_adjudication import (
    Stage2RoleAdjudicationConfig,
    adjudicate_stage2_roles,
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
        selection_mode="multi_model",
        internal_cv_folds=2,
        max_iter=1500,
        minimum_log10_alpha=-1.5,
        maximum_log10_alpha=0,
        optimization_tolerance=1e-4,
        categorical_min_count=2,
        multi_model=Stage2MultiModelConfig(
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
def test_real_models_are_honest_cover_candidates_and_resume_without_refitting(
    tmp_path, monkeypatch, binary
):
    arguments = sample_inputs()
    if binary:
        arguments["outcome_type"] = "binary"
        arguments["dataset"]["outcome"] = (
            arguments["dataset"]["outcome"] > arguments["dataset"]["outcome"].median()
        ).astype(int)
        arguments["extracted_fit"]["other"] = np.where(
            arguments["extracted_fit"]["other"] > 0, "a", "b"
        )
        arguments["definitions"][-1].update(value_type="categorical", categories_or_unit=["a", "b"])
    result = select_stage2_features_elastic_net(**arguments, checkpoint_dir=tmp_path)
    _, report, dependencies, latents = result
    assert not latents and len(dependencies) == 4
    assert set(report["evaluable_cells_by_family"]) == set(FAMILIES)
    assert all(v > 0 for v in report["evaluable_cells_by_family"].values())
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
    resumed = select_stage2_features_elastic_net(
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
        select_stage2_features_elastic_net(**arguments, checkpoint_dir=tmp_path)
    arguments = sample_inputs()
    arguments["dataset"]["treatment"] = 0
    with pytest.raises(ValueError, match="both binary treatment arms"):
        select_stage2_features_elastic_net(**arguments, checkpoint_dir=tmp_path)


def test_all_modifier_evidence_uses_overlap_while_associations_keep_all_rows(monkeypatch):
    from oci.inference import stage2_multi_model_selection as selection

    arguments = sample_inputs()
    arguments["policy"] = replace(
        arguments["policy"],
        min_propensity=0.1,
        max_propensity=0.9,
        multi_model=replace(arguments["policy"].multi_model, repeats=1),
    )
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
        },
    )
    _, original, _, _ = select_stage2_features_elastic_net(**arguments)
    excluded = ~selection.linear.propensity_eligibility(p(arguments["extracted_fit"]), 0.1, 0.9)
    changed = arguments["dataset"].copy()
    changed.loc[np.flatnonzero(excluded), "outcome"] += 100
    _, poisoned, _, _ = select_stage2_features_elastic_net(**{**arguments, "dataset": changed})
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
    )
    assert all(r["status"] == "not_evaluable" for r in empty["records"] if r["role"] == "effect")


def test_subsets_and_group_permutation_preserve_candidate_units():
    for repeat in (0, 1, 2):
        groups = candidate_subsets(list("abcdefg"), repeat=repeat, subset_size=3, seed=55)
        assert sorted(k for group in groups for k in group) == list("abcdefg")
        assert len(groups) == (1 if repeat == 0 else 3)
    x = np.array([[1, 0, 1], [0, 1, 2], [1, 0, 3], [0, 1, 4]], dtype=float)
    calls = []

    def score(values):
        assert np.all(values[:, :2].sum(axis=1) == 1)
        calls.append(values.copy())
        return float(np.mean(values[:, 0] * values[:, 2]))

    _, result = grouped_permutation_gain(
        x, ["category", "category", "number"], score, repeats=3, seed=24
    )
    assert set(result) == {"category", "number"} and len(calls) == 7
    assert np.array_equal(x, [[1, 0, 1], [0, 1, 2], [1, 0, 3], [0, 1, 4]])


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


def test_themes_cover_all_batches_roles_cite_evidence_and_resume(tmp_path):
    definitions, report = role_fixture()
    policy = Stage2RoleAdjudicationConfig(max_candidates_per_request=2)
    calls = []

    def request(messages, validate, **kwargs):
        payload = json.loads(messages[1]["content"])
        assert "DO_NOT_LEAK" not in json.dumps(messages)
        calls.append(payload["task"])
        if "themes" in payload["task"]:
            if "candidates" in payload:
                members = [c["feature_id"] for c in payload["candidates"]]
                citations = [
                    r["evidence_id"] for c in payload["candidates"] for r in c["modeling_evidence"]
                ]
            else:
                members = list(
                    dict.fromkeys(k for t in payload["themes"] for k in t["member_feature_ids"])
                )
                citations = list(
                    dict.fromkeys(k for t in payload["themes"] for k in t["evidence_ids"])
                )
            return validate(
                {
                    "themes": [
                        {
                            "name": "Shared evidence",
                            "member_feature_ids": members,
                            "evidence_ids": citations[:12],
                            "interpretation": "Candidate-specific support.",
                            "disagreements": "Shared samples limit independence.",
                        }
                    ]
                }
            )
        decisions = []
        for c in payload["candidates"]:
            decisions.append(
                {
                    "feature_id": c["feature_id"],
                    "roles": ["confounder"] if c["feature_id"] == "f0" else ["effect_modifier"],
                    "evidence_ids": [r["evidence_id"] for r in c["modeling_evidence"]],
                    "evidence_for": ["Supplied support."],
                    "evidence_against": [],
                    "inner_fold_consistency": "Only one fold supplied.",
                    "cross_method_reconciliation": "Only one family supplied.",
                    "rationale": "Exploratory support.",
                    "stability": "insufficient",
                }
            )
        assert payload["themes"]
        return validate({"summary": "Provisional roles.", "decisions": decisions})

    selected, audit, evidence = adjudicate_stage2_roles(
        definitions=definitions,
        statistical_report=report,
        request_json=request,
        output_dir=tmp_path,
        policy=policy,
    )
    assert len(selected) == 7 and selected[0]["roles"] == ["confounder"]
    assert {k for t in audit["themes"] for k in t["member_feature_ids"]} == {
        f"f{i}" for i in range(7)
    }
    assert "merge_stage2_multi_model_themes" in calls
    assert calls.count("adjudicate_stage2_multi_model_roles") == 4
    first_count = len(calls)
    second = adjudicate_stage2_roles(
        definitions=definitions,
        statistical_report=report,
        request_json=request,
        output_dir=tmp_path,
        policy=policy,
    )
    assert len(calls) == first_count and second == (selected, audit, evidence)
    response = tmp_path / "roles/batch_000/response.json"
    response.write_text(response.read_text().replace("Exploratory support.", "Tampered evidence."))
    with pytest.raises(ValueError, match="corrupt"):
        adjudicate_stage2_roles(
            definitions=definitions,
            statistical_report=report,
            request_json=request,
            output_dir=tmp_path,
            policy=policy,
        )


def test_role_review_rejects_lost_candidates_and_invented_citations():
    validator = _themes_validator(["a", "b"], ["e1"], maximum=2)
    with pytest.raises(ValueError, match="preserve every"):
        validator(
            {
                "themes": [
                    {
                        "name": "A",
                        "member_feature_ids": ["a"],
                        "evidence_ids": ["e1"],
                        "interpretation": "A",
                        "disagreements": "B",
                    }
                ]
            }
        )
    definitions, report = role_fixture()
    evidence = build_multi_model_role_evidence(
        definitions=definitions, statistical_report=report, policy=Stage2RoleAdjudicationConfig()
    )
    validate = _decision_validator(definitions[:1], evidence["candidates"][:1])
    with pytest.raises(ValueError, match="supplied evidence"):
        validate(
            {
                "summary": "A",
                "decisions": [
                    {
                        "feature_id": "f0",
                        "roles": ["confounder"],
                        "evidence_ids": ["invented"],
                        "rationale": "A",
                        "inner_fold_consistency": "A",
                        "cross_method_reconciliation": "A",
                        "stability": "mixed",
                    }
                ],
            }
        )


def test_mode_is_explicit_and_legacy_checkpoint_policies_are_unchanged():
    legacy = Stage2ElasticNetSelectionConfig().public_dict()
    assert "selection_mode" not in legacy and "multi_model" not in legacy
    independent = Stage2ElasticNetSelectionConfig(selection_mode="independent_tasks").public_dict()
    assert "multi_model" not in independent
    new = statistical_selection_config_from_mapping(
        {"selection_mode": "multi_model", "multi_model": {"repeats": 2}}
    )
    assert new.multi_model.repeats == 2
    assert statistical_selection_config_from_mapping(new.public_dict()) == new
    with pytest.raises(ValueError, match="unsupported"):
        statistical_selection_config_from_mapping(
            {"selection_mode": "multi_model", "multi_model": {"unknown": True}}
        )
    with pytest.raises(ValueError):
        replace(new, multi_model=replace(new.multi_model, row_fraction=float("nan"))).validate()


def test_new_example_is_active_requires_llm_and_freezes_resume_policy(tmp_path):
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
    assert config.stage2.statistical_selection.selection_mode == "multi_model"
    assert config.stage2.statistical_selection.multi_model.modifier_count.enabled
    assert (config.stage2.min_propensity, config.stage2.max_propensity) == (0.1, 0.9)
    raw["stage2"]["role_adjudication"]["enabled"] = False
    with pytest.raises(ValueError, match="requires role_adjudication"):
        compile_config(raw, config_dir=path.parent)
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
    old = json.loads(json.dumps(saved))
    for version in ("stage2_multi_model_selection_v1", "stage2_multi_model_selection_v2"):
        old["statistical_selection"]["multi_model"]["schema_version"] = version
        (root / "config.json").write_text(json.dumps(old))
        with pytest.raises(RuntimeError, match="automatic modifier count"):
            validate_selection_resume(SimpleNamespace(stage2=config.stage2, output_dir=tmp_path))
    (root / "config.json").write_text(json.dumps(saved))
    changed_trees = replace(config.stage2, estimation_trees=config.stage2.estimation_trees * 2)
    with pytest.raises(RuntimeError, match="modifier-count estimation_trees changed"):
        validate_selection_resume(SimpleNamespace(stage2=changed_trees, output_dir=tmp_path))
    assert _stage2_reselection_policy_fingerprint(config) != _stage2_reselection_policy_fingerprint(
        replace(config, stage2=changed_trees)
    )
    changed = replace(
        config.stage2,
        statistical_selection=replace(
            config.stage2.statistical_selection,
            multi_model=replace(config.stage2.statistical_selection.multi_model, repeats=7),
        ),
    )
    with pytest.raises(RuntimeError, match="multi-model policy changed"):
        validate_selection_resume(SimpleNamespace(stage2=changed, output_dir=tmp_path))
