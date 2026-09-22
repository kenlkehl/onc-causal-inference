"""Nested ranking, fixed scoring populations, role protection, and resume integrity."""

from dataclasses import replace
import json

import numpy as np
import pytest

from oci.inference import stage2_modifier_count as count
from oci.inference import stage2_modifier_ranking as ranking
from oci.inference import stage2_multi_model_selection as numerical
from oci.inference.stage2_multi_model_config import ModifierCountConfig
from oci.inference.stage2_elastic_net_selection import (
    statistical_selection_config_from_mapping,
)
from oci.inference.stage2_role_adjudication import Stage2RoleAdjudicationConfig
from tests.test_stage2_multi_model import sample_inputs, role_fixture


def ranking_request(messages, validate, **kwargs):
    assert "DO_NOT" not in json.dumps(messages)
    payload = json.loads(messages[1]["content"])
    # A deterministic stand-in for an LLM ordering; the numerical fits are real.
    priority = {"m": 0, "c": 1, "noise": 2, "other": 3}
    cards = sorted(
        payload["candidates"],
        key=lambda c: (priority.get(c["feature_id"], 10), c["feature_id"]),
    )
    return validate(
        {
            "ranking": [
                {
                    "feature_id": c["feature_id"],
                    "evidence_ids": [
                        r["evidence_id"]
                        for r in c["modeling_evidence"]
                        if r["role"] == "effect"
                    ],
                    "rationale": "Compare supplied effect evidence.",
                }
                for c in cards
            ]
        }
    )


def test_count_rule_and_configuration_keep_legacy_policies_separate():
    losses = {0: [1.2, 1.3, 1.2], 8: [0.99, 1.02, 1.00], 16: [1.0, 1.0, 1.0]}
    assert (
        count.choose_modifier_count(losses, rule="minimum_r_loss")[
            "chosen_additional_count"
        ]
        == 16
    )
    assert (
        count.choose_modifier_count(losses, rule="one_standard_error")[
            "chosen_additional_count"
        ]
        == 8
    )
    assert (
        count.choose_modifier_count(
            {0: [1.0, 1.0], 4: [1.0, 1.0]}, rule="minimum_r_loss"
        )["chosen_additional_count"]
        == 0
    )
    with pytest.raises(ValueError, match="two finite"):
        count.choose_modifier_count({0: [1.0]}, rule="minimum_r_loss")
    with pytest.raises(ValueError, match="identical"):
        count.choose_modifier_count(
            {0: [1.0, 1.0], 4: [1.0, 1.0, 1.0]}, rule="minimum_r_loss"
        )
    config = statistical_selection_config_from_mapping(
        {"selection_mode": "multi_model"}
    )
    assert config.multi_model.modifier_count.enabled
    assert config.multi_model.modifier_count.selection_rule == "minimum_r_loss"
    assert statistical_selection_config_from_mapping(config.public_dict()) == config
    disabled = replace(
        config,
        multi_model=replace(
            config.multi_model, modifier_count=ModifierCountConfig(enabled=False)
        ),
    )
    assert statistical_selection_config_from_mapping(disabled.public_dict()) == disabled
    assert (
        "multi_model" not in statistical_selection_config_from_mapping({}).public_dict()
    )
    for change in (
        {"candidate_counts": [4, 0]},
        {"candidate_counts": [0, True]},
        {"selection_rule": "oracle"},
        {"enabled": "yes"},
    ):
        with pytest.raises(ValueError):
            statistical_selection_config_from_mapping(
                {
                    "selection_mode": "multi_model",
                    "multi_model": {"modifier_count": change},
                }
            )


def test_bounded_ranking_covers_candidates_preserves_locks_and_detects_corruption(
    tmp_path,
):
    definitions, report = role_fixture()
    policy = Stage2RoleAdjudicationConfig(max_candidates_per_request=2)
    calls, seen = [], set()

    def request(messages, validate, **kwargs):
        payload = json.loads(messages[1]["content"])
        assert len(payload["candidates"]) <= 2
        seen.update(c["feature_id"] for c in payload["candidates"])
        calls.append(payload["task"])
        return ranking_request(messages, validate, **kwargs)

    arguments = dict(
        definitions=definitions,
        statistical_report=report,
        role_policy=policy,
        maximum=3,
        maximum_chars=100_000,
        model_identity="controlled-response",
        output_dir=tmp_path,
    )
    first = ranking.rank_modifier_candidates(**arguments, request_json=request)
    assert [r["feature_id"] for r in first["ranking"]] == ["f1", "f2", "f3"]
    assert (
        seen == {f"f{i}" for i in range(1, 7)} and "f0" in first["locked_feature_ids"]
    )
    assert "merge_stage2_modifier_ranking" in calls
    resumed = ranking.rank_modifier_candidates(
        **arguments,
        request_json=lambda *a, **k: pytest.fail("cached ranking requested again"),
    )
    assert resumed == first
    path = next(tmp_path.glob("requests/*/response.json"))
    cached = json.loads(path.read_text())
    cached["result"]["ranking"][0]["rationale"] = "changed"
    path.write_text(json.dumps(cached))
    with pytest.raises(ValueError, match="corrupt"):
        ranking.rank_modifier_candidates(**arguments, request_json=request)


def test_rank_validator_rejects_foreign_evidence_and_invalid_merges():
    definitions, report = role_fixture()
    evidence = ranking.build_multi_model_role_evidence(
        definitions=definitions,
        statistical_report=report,
        policy=Stage2RoleAdjudicationConfig(),
    )
    cards = evidence["candidates"][1:3]
    refs = {
        c["feature_id"]: [
            r["evidence_id"] for r in c["modeling_evidence"] if r["role"] == "effect"
        ]
        for c in cards
    }
    rows = [
        {"feature_id": k, "evidence_ids": v, "rationale": "Evidence."}
        for k, v in refs.items()
    ]
    validate = ranking._ranking_validator(cards, [["f1", "f2"]])
    with pytest.raises(ValueError, match="preserve order"):
        validate({"ranking": rows[::-1]})
    rows[0]["evidence_ids"] = rows[1]["evidence_ids"]
    with pytest.raises(ValueError, match="own supplied"):
        validate({"ranking": rows})


def test_zero_budget_preserves_confounders_and_exact_investigator_roles():
    arguments = sample_inputs()
    definitions = arguments["definitions"]
    definitions[-1].update(configured_explicit_feature=True, roles=["effect_modifier"])
    selected = [
        {
            **f,
            "roles": ["confounder", "effect_modifier"]
            if f["name"] == "c"
            else ["effect_modifier"],
        }
        for f in definitions
    ]
    report = {
        "decisions": [
            {"feature_id": f["feature_id"], "roles": f["roles"]} for f in selected
        ]
    }
    ordered = [
        {"feature_id": "m", "evidence_ids": ["m-effect"], "rationale": "Evidence."}
    ]
    kept, reviewed = count._apply_budget(
        definitions, selected, report, ordered, 0, ["other"]
    )
    assert {f["feature_id"]: f["roles"] for f in kept} == {
        "c": ["confounder"],
        "other": ["effect_modifier"],
    }
    assert kept[0]["nuisance_model_roles"] == ["treatment", "outcome"]
    assert len(reviewed["decisions"]) == 4
    assert report["decisions"][0]["roles"] == ["confounder", "effect_modifier"]
    train = arguments["extracted_fit"].iloc[:40]
    valid = arguments["extracted_fit"].iloc[40:60]
    tr = np.tile([-0.5, 0.5], 20)
    result = count._score_prefix(
        train=train,
        valid=valid,
        definitions=definitions,
        feature_ids=[],
        tr=tr,
        yr=2 * tr,
        tvr=np.tile([-0.5, 0.5], 10),
        yvr=np.zeros(20),
        seed=11,
        trees=12,
    )
    assert result["constant_design"] and result["predictions"] == [2.0] * 20
    assert result["r_loss"] == 1.0


@pytest.mark.parametrize("binary", [False, True])
def test_real_nested_selection_scoring_alignment_and_no_refit_resume(
    tmp_path, monkeypatch, binary
):
    arguments = sample_inputs()
    if binary:
        arguments["dataset"]["outcome"] = (arguments["dataset"]["outcome"] > 0).astype(
            int
        )
        arguments["outcome_type"] = "binary"
        arguments["extracted_fit"]["other"] = np.where(
            arguments["extracted_fit"]["other"] > 0, "a", "b"
        )
        arguments["definitions"][-1].update(
            value_type="categorical", categories_or_unit=["a", "b"]
        )
    cfg = ModifierCountConfig(
        candidate_counts=(0, 1, 2), max_ranked_modifiers=3, forest_seeds=2
    )
    arguments["policy"] = replace(
        arguments["policy"],
        multi_model=replace(
            arguments["policy"].multi_model, repeats=1, modifier_count=cfg
        ),
    )
    numerical_root = tmp_path / "numerical"
    _, report, _, _ = numerical.select_stage2_features_multi_model(
        **arguments, checkpoint_dir=numerical_root
    )
    selected = [
        {
            **f,
            "roles": ["confounder", "effect_modifier"]
            if f["name"] == "c"
            else ["effect_modifier"],
        }
        for f in arguments["definitions"]
    ]
    role_report = {
        "decisions": [
            {"feature_id": f["feature_id"], "roles": f["roles"]} for f in selected
        ]
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
        return numerical.select_stage2_features_multi_model(**values)

    extras = dict(
        selected=selected,
        role_report=role_report,
        statistical_report=report,
        request_json=ranking_request,
        role_policy=Stage2RoleAdjudicationConfig(max_candidates_per_request=2),
        output_dir=tmp_path / "count",
        numerical_checkpoint_dir=numerical_root,
        model_identity="controlled-response",
        estimation_trees=12,
        run_numerical=run_nested,
    )
    result = count.select_modifier_count(**arguments, **extras)
    kept, roles, audit = result
    assert (
        len(nested_calls) == 2
        and audit["boundaries"][
            "ranking_and_numerical_evidence_nested_within_count_training"
        ]
    )
    assert next(f for f in kept if f["feature_id"] == "c")["roles"][0] == "confounder"
    assert (
        sum("effect_modifier" in f["roles"] for f in kept)
        == audit["chosen_modifier_count"]
    )
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
        all_ids = []
        for path in (root / f"fold_{fold:03d}").glob("size_*/seed_*.json"):
            cell = json.loads(path.read_text())["result"]
            assert cell["validation_row_ids"] == valid_ids
            expected = (
                labels.loc[valid_ids, "outcome"].to_numpy()
                - m
                - (labels.loc[valid_ids, "treatment"].to_numpy() - e)
                * np.asarray(cell["predictions"])
            ) ** 2
            assert np.allclose(cell["squared_errors"], expected)
            assert np.isclose(cell["r_loss"], expected.mean())
            all_ids.append(cell["validation_row_ids"])
        assert (
            len(all_ids) == 8
        )  # Four sizes, two seeds; one common scoring population.
    changed = arguments["dataset"].copy()
    changed.loc[96:, ["treatment", "outcome"]] = 1e12
    changed["oracle"] = "DO_NOT_READ_CHANGED"
    monkeypatch.setattr(
        count, "_score_prefix", lambda **k: pytest.fail("cached forest refitted")
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
            modifier_count=replace(cfg, selection_rule="one_standard_error"),
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
    saved = root / "result.json"
    corrupted = json.loads(saved.read_text())
    corrupted["result"]["report"]["chosen_modifier_count"] += 1
    saved.write_text(json.dumps(corrupted))
    with pytest.raises(ValueError, match="corrupt"):
        count.select_modifier_count(**arguments, **extras)
