"""The final interaction model leaves the test-selected modifiers unpenalized.

Under ``modifier_selection.rule: orthogonal_tests`` the selected modifiers'
interaction and main-effect columns are exempt from the tuned group elastic
net, guarded by events per parameter, with a fixed-ridge and then a fully
penalized fallback for separation. Zero modifiers, a failed guard, the
``balanced_evidence`` rule, and every validation fit keep the single tuned
penalty exactly.
"""

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy.special import expit

from oci.inference import plain_handoff_stage2_analysis as analysis
from oci.inference import stage2_effect_estimators as effects
from oci.inference import stage2_elastic_net_selection as linear
from oci.inference import stage2_modifier_count as count
from oci.inference import stage2_multi_model_selection as numerical
from oci.inference.stage2_elastic_net_selection import (
    statistical_selection_config_from_mapping,
)
from oci.inference.stage2_multi_model_config import ModifierSelectionConfig
from tests.test_stage2_final_modeling import candidates as final_candidates
from tests.test_stage2_final_modeling import inputs as final_inputs
from tests.test_stage2_multi_model import feature, small_policy
from tests.test_stage2_treatment_penalty import strong_policy

UNPENALIZED = effects.UnpenalizedSelectedModifiers(events_per_parameter=10.0)


def _with_selection(policy, **settings):
    multi = policy.multi_model
    return replace(
        policy,
        multi_model=replace(
            multi, modifier_selection=replace(multi.modifier_selection, **settings)
        ),
    )


def _data(binary, *, n=600, seed=7):
    rng = np.random.default_rng(seed)
    c = rng.normal(size=(n, 3))
    m = rng.normal(size=n)
    level = rng.choice(["a", "b", "c"], size=n)
    t = rng.binomial(1, expit(0.4 * c[:, 0]))
    base = -0.2 + 0.5 * c.sum(axis=1) + 0.3 * m + 0.4 * (level == "b")
    shift = 0.3 + 0.8 * m - 0.6 * (level == "c")
    y = (
        rng.binomial(1, expit(base + t * shift)).astype(float)
        if binary
        else base + t * shift + rng.normal(scale=0.5, size=n)
    )
    frame = pd.DataFrame(
        {"_oci_row_id": range(n), "c0": c[:, 0], "c1": c[:, 1], "c2": c[:, 2], "m": m,
         "level": level}
    )
    definitions = [
        feature("c0"), feature("c1"), feature("c2"), feature("m"),
        {**feature("level"), "value_type": "categorical", "categories_or_unit": ["a", "b", "c"]},
    ]
    return frame, definitions, t, y


def _fit(binary, *, modifiers=("m", "level"), policy=None, frame_data=None, **kwargs):
    frame, definitions, t, y = frame_data or _data(binary)
    cut = int(0.8 * len(frame))
    return effects.fit_interaction_effect(
        train=frame.iloc[:cut], valid=frame.iloc[cut:], definitions=definitions,
        modifier_ids=list(modifiers), treatment=t[:cut], outcome=y[:cut], binary=binary,
        policy=policy or small_policy(), seed=17, **kwargs,
    )


@pytest.mark.parametrize("binary", [False, True])
def test_selected_modifier_groups_are_exempt_in_every_fit_and_confounders_stay_penalized(
    monkeypatch, binary
):
    original = linear._fit_group_elastic_net
    structures = []

    def recording(x, y, **kwargs):
        structures.append(kwargs["groups"])
        return original(x, y, **kwargs)

    monkeypatch.setattr(linear, "_fit_group_elastic_net", recording)
    frame, definitions, t, y = _data(binary)
    cut = 480
    adjustment = effects.nuisance_adjustment(
        train_propensity=expit(0.4 * frame.c0[:cut]), valid_propensity=expit(0.4 * frame.c0[cut:]),
        train_mu0=np.full(cut, 0.4) + 0.01 * frame.c1[:cut],
        valid_mu0=np.full(len(frame) - cut, 0.4) + 0.01 * frame.c1[cut:], binary=binary,
    )
    result = _fit(binary, frame_data=(frame, definitions, t, y), adjustment=adjustment,
                  unpenalized_selected=UNPENALIZED)
    exempt = {
        "treatment", "adjustment:logit_propensity",
        "adjustment:logit_prognostic_mu0" if binary else "adjustment:prognostic_mu0",
        "main:m", "effect:m", "main:level", "effect:level",
    }
    policy = small_policy()
    grid = policy.multi_model.regularization_grid_size
    assert len(structures) == policy.internal_cv_folds * grid + 1
    for groups in structures:
        penalties = dict(zip(groups.feature_ids, groups.penalty_factors))
        assert {key for key, value in penalties.items() if value == 0} == exempt
        assert {key for key, value in penalties.items() if value == 1} == {
            "main:c0", "main:c1", "main:c2"}
        assert groups.fixed_ridge is None
    audit = result["audit"]
    record = audit["selected_modifier_penalty"]
    assert record["path"] == "unpenalized" and record["reason"] is None
    assert record["unpenalized_groups"] == ["main:level", "effect:level", "main:m", "effect:m"]
    assert set(audit["unpenalized_groups"]) == exempt
    assert audit["selected_modifier_groups"] == {
        "groups": record["unpenalized_groups"], "penalty": "none", "fixed_ridge": None}
    assert record["guard"]["passed"] and record["attempts"][0]["accepted"]


@pytest.mark.parametrize("binary", [False, True])
def test_strong_penalty_leaves_an_unpenalized_glm_on_treatment_and_modifiers(binary):
    """Confounders are zeroed; T, Z and T x Z equal the unpenalized GLM fit."""
    frame, definitions, t, y = _data(binary, n=800, seed=3)
    policy = replace(strong_policy(), max_iter=20_000, optimization_tolerance=1e-10)
    result = _fit(binary, modifiers=["m"], policy=policy, frame_data=(frame, definitions, t, y),
                  unpenalized_selected=UNPENALIZED)
    audit = result["audit"]
    coefficients = dict(zip(
        [f"{block}:{key}" for key, block in
         zip(audit["coefficient_feature_ids"], audit["coefficient_blocks"])],
        audit["coefficients"],
    ))
    for key in ("main:c0", "main:c1", "main:c2", "main:level"):
        assert coefficients[key] == 0.0
    train, valid = frame.iloc[:640], frame.iloc[640:]
    design = np.column_stack((np.ones(640), t[:640], train.m, t[:640] * train.m))
    family = sm.families.Binomial() if binary else sm.families.Gaussian()
    reference = sm.GLM(y[:640], design, family=family).fit(tol=1e-12)
    arms = [
        reference.predict(np.column_stack((np.ones(160), np.full(160, arm), valid.m,
                                           arm * valid.m)))
        for arm in (0, 1)
    ]
    np.testing.assert_allclose(result["tau"], arms[1] - arms[0], atol=2e-5)
    # The same data with the single tuned penalty shrinks the interaction to zero.
    shrunk = _fit(binary, modifiers=["m"], policy=policy, frame_data=(frame, definitions, t, y))
    effect = [v for k, b, v in zip(shrunk["audit"]["coefficient_feature_ids"],
                                   shrunk["audit"]["coefficient_blocks"],
                                   shrunk["audit"]["coefficients"]) if b == "effect"]
    assert effect == [0.0]


@pytest.mark.parametrize("binary", [False, True])
def test_zero_selected_modifiers_equal_the_fully_penalized_model_exactly(binary):
    plain = _fit(binary, modifiers=[])
    requested = _fit(binary, modifiers=[], unpenalized_selected=UNPENALIZED)
    np.testing.assert_array_equal(requested["tau"], plain["tau"])
    np.testing.assert_array_equal(requested["mu0"], plain["mu0"])
    record = requested["audit"].pop("selected_modifier_penalty")
    assert requested["audit"] == plain["audit"]
    assert (record["path"], record["reason"]) == ("fully_penalized", "no_selected_modifiers")
    assert record["guard"]["unpenalized_parameters"] == 0 and record["attempts"] == []


@pytest.mark.parametrize("binary", [False, True])
def test_events_per_parameter_guard_falls_back_to_the_fully_penalized_model(binary):
    frame, definitions, t, y = _data(binary)
    plain = _fit(binary, frame_data=(frame, definitions, t, y))
    allowed = _fit(binary, frame_data=(frame, definitions, t, y), unpenalized_selected=UNPENALIZED)
    guard = allowed["audit"]["selected_modifier_penalty"]["guard"]
    # m: value; level: two reference contrasts. Main effect plus interaction each.
    assert guard["modifier_encoded_columns"] == {"level": 2, "m": 1}
    assert guard["unpenalized_parameters"] == 6
    assert guard["eligible_rows"] == 480
    if binary:
        events = int(y[:480].sum())
        assert (guard["events"], guard["non_events"]) == (events, 480 - events)
        assert guard["information"] == min(events, 480 - events)
        assert guard["information_basis"] == "min_events_non_events"
    else:
        assert guard["information"] == 480 and guard["information_basis"] == "eligible_rows"
    assert guard["parameter_limit"] == pytest.approx(guard["information"] / 10.0)
    # Just above the limit: the guard fails and the model is the fully penalized one.
    ratio = guard["information"] / 5.9
    blocked = _fit(binary, frame_data=(frame, definitions, t, y),
                   unpenalized_selected=effects.UnpenalizedSelectedModifiers(ratio))
    record = blocked["audit"].pop("selected_modifier_penalty")
    assert (record["path"], record["reason"]) == ("fully_penalized", "events_per_parameter_guard")
    assert record["guard"]["parameter_limit"] == pytest.approx(5.9)
    assert not record["guard"]["passed"] and record["attempts"] == []
    np.testing.assert_array_equal(blocked["tau"], plain["tau"])
    assert blocked["audit"] == plain["audit"]
    exactly = _fit(binary, frame_data=(frame, definitions, t, y),
                   unpenalized_selected=effects.UnpenalizedSelectedModifiers(
                       guard["information"] / 6.0))
    assert exactly["audit"]["selected_modifier_penalty"]["path"] == "unpenalized"


def _separated(n=800, rare=14, seed=2):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4))
    level = rng.choice(["a", "b"], size=n)
    level[rng.choice(n, size=rare, replace=False)] = "c"
    t = rng.binomial(1, 0.5, size=n)
    y = rng.binomial(1, expit(-0.4 + 0.3 * x.sum(axis=1) + t * (0.4 * (level == "b") - 0.3)))
    c = level == "c"
    y[c] = t[c]  # The rare level separates treated from control patients.
    frame = pd.DataFrame(x, columns=[f"x{i}" for i in range(4)])
    frame["level"] = level
    frame.insert(0, "_oci_row_id", range(n))
    definitions = [feature(f"x{i}") for i in range(4)] + [
        {**feature("level"), "value_type": "categorical", "categories_or_unit": ["a", "b", "c"]}]
    return frame, definitions, t, y.astype(float)


def test_separation_falls_back_to_the_fixed_ridge_and_is_audited(monkeypatch):
    original = linear._fit_group_elastic_net
    ridges = []

    def recording(x, y, **kwargs):
        ridges.append(kwargs["groups"].fixed_ridge)
        return original(x, y, **kwargs)

    monkeypatch.setattr(linear, "_fit_group_elastic_net", recording)
    data = _separated()
    policy = linear.Stage2ElasticNetSelectionConfig()
    result = _fit(True, modifiers=["level"], policy=policy, frame_data=data,
                  unpenalized_selected=UNPENALIZED)
    record = result["audit"]["selected_modifier_penalty"]
    assert record["path"] == "separation_ridge"
    assert record["reason"] == "unpenalized_fit_separated_or_did_not_converge"
    assert [attempt["path"] for attempt in record["attempts"]] == [
        "unpenalized", "separation_ridge"]
    assert record["attempts"][0] == {
        "path": "unpenalized", "accepted": False, "reason": "final_penalized_fit_did_not_converge"}
    assert record["attempts"][1]["accepted"]
    assert record["attempts"][1]["max_range_effect"] <= effects.SELECTED_MODIFIER_MAX_RANGE_EFFECT
    assert record["fixed_ridge_groups"] == ["main:level", "effect:level"]
    assert record["unpenalized_groups"] == []
    assert record["fixed_ridge"] == effects.SELECTED_MODIFIER_SEPARATION_RIDGE == 0.01
    assert result["audit"]["selected_modifier_groups"]["penalty"] == "fixed_ridge"
    assert "main:level" not in result["audit"]["unpenalized_groups"]
    # The ridge fits carry the fixed ridge on exactly the two modifier groups.
    final = ridges[-1]
    assert final is not None and sorted(final[final > 0]) == [0.01, 0.01]
    assert np.isfinite(result["tau"]).all()
    summary = effects.selected_modifier_penalty_summary(result["audit"])
    assert summary["path"] == "separation_ridge"
    assert summary["attempted_paths"] == ["unpenalized", "separation_ridge"]


def test_range_effect_check_rejects_both_paths_and_uses_the_penalized_model():
    data = _separated(rare=0, seed=5)
    strict = effects.UnpenalizedSelectedModifiers(events_per_parameter=10.0, max_range_effect=1e-9)
    result = _fit(True, modifiers=["level"], frame_data=data, unpenalized_selected=strict)
    record = result["audit"].pop("selected_modifier_penalty")
    assert (record["path"], record["reason"]) == ("fully_penalized", "separation_or_nonconvergence")
    assert [a["reason"] for a in record["attempts"]] == [
        "selected_modifier_range_effect_exceeds_limit"] * 2
    plain = _fit(True, modifiers=["level"], frame_data=data)
    np.testing.assert_array_equal(result["tau"], plain["tau"])
    assert result["audit"] == plain["audit"]


def test_fixed_ridge_prox_and_default_structure():
    values = np.array([2.0, -3.0, 4.0, -5.0])
    ids = ["penalized", "ridge", "ridge", "free"]
    kwargs = {"step": 0.5, "regularization": 100.0, "group_ratio": 0.8}
    default = linear._group_structure(ids, unpenalized_feature_ids=["ridge", "free"])
    assert default.fixed_ridge is None
    ridged = linear._group_structure(
        ids, unpenalized_feature_ids=["ridge", "free"], fixed_ridge={"ridge": 0.2})
    np.testing.assert_array_equal(ridged.fixed_ridge, [0.0, 0.2, 0.0])
    plain = linear._group_ridge_prox(values, groups=default, **kwargs)
    actual = linear._group_ridge_prox(values, groups=ridged, **kwargs)
    np.testing.assert_array_equal(actual[[0, 3]], plain[[0, 3]])
    np.testing.assert_allclose(actual[1:3], values[1:3] / (1 + 0.5 * 0.2))
    np.testing.assert_array_equal(plain[1:], values[1:])


def test_fit_linear_rejects_inconsistent_exemptions():
    frame, definitions, t, y = _data(False, n=60)
    common = {"binary": False, "treatment": t.astype(float), "valid_treatment": t.astype(float),
                  "policy": small_policy(), "seed": 1, "ratio": 0.8}
    with pytest.raises(ValueError, match="only to interaction models"):
        numerical._fit_linear(frame, frame, definitions, y, kind="main",
                              unpenalized_modifier_ids={"m"}, **common)
    with pytest.raises(ValueError, match="must be interaction modifiers"):
        numerical._fit_linear(frame, frame, definitions, y, kind="interactions",
                              modifier_ids={"level"}, unpenalized_modifier_ids={"m"}, **common)
    with pytest.raises(ValueError, match="fixed modifier ridge"):
        numerical._fit_linear(frame, frame, definitions, y, kind="interactions",
                              modifier_ids={"m"}, modifier_ridge=0.01, **common)


def test_configuration_validation_resolution_and_identity_views():
    default = statistical_selection_config_from_mapping(None)
    chosen = default.multi_model.modifier_selection
    assert chosen.unpenalized_selected_interactions is None
    assert chosen.resolved_unpenalized_selected_interactions is True
    assert chosen.resolved_unpenalized_events_per_parameter("binary") == 10.0
    assert chosen.resolved_unpenalized_events_per_parameter("continuous") == 15.0
    # Null keys are not serialized, so saved default policies are unchanged.
    assert "unpenalized_selected_interactions" not in chosen.public_dict()
    assert "unpenalized_events_per_parameter" not in chosen.public_dict()
    assert default.selection_dict() == default.public_dict()

    def parse(**settings):
        return statistical_selection_config_from_mapping(
            {"multi_model": {"modifier_selection": settings}})

    explicit = parse(unpenalized_selected_interactions=True, unpenalized_events_per_parameter=20)
    explicit_choice = explicit.multi_model.modifier_selection
    assert explicit_choice.resolved_unpenalized_events_per_parameter("continuous") == 20.0
    assert explicit.public_dict()["multi_model"]["modifier_selection"][
        "unpenalized_events_per_parameter"] == 20
    assert explicit.selection_dict() == default.selection_dict()
    assert statistical_selection_config_from_mapping(explicit.public_dict()) == explicit
    off = parse(unpenalized_selected_interactions=False)
    assert off.multi_model.modifier_selection.resolved_unpenalized_selected_interactions is False
    assert off.multi_model.modifier_selection.resolved_unpenalized_events_per_parameter(
        "binary") is None
    assert off.selection_dict() == default.selection_dict()
    assert off.public_dict() != default.public_dict()
    balanced = parse(rule="balanced_evidence")
    balanced_choice = balanced.multi_model.modifier_selection
    assert balanced_choice.resolved_unpenalized_selected_interactions is False
    parse(rule="balanced_evidence", unpenalized_selected_interactions=False)
    with pytest.raises(ValueError, match="null or false under balanced_evidence"):
        parse(rule="balanced_evidence", unpenalized_selected_interactions=True)
    for value in (0, -1, float("nan"), float("inf"), True, "10"):
        with pytest.raises(ValueError, match="positive finite number"):
            parse(unpenalized_events_per_parameter=value)
    for settings in ({"rule": "balanced_evidence"}, {"unpenalized_selected_interactions": False}):
        with pytest.raises(ValueError, match="must be null unless"):
            parse(unpenalized_events_per_parameter=10, **settings)
    with pytest.raises(ValueError, match="null or boolean"):
        parse(unpenalized_selected_interactions="yes")
    with pytest.raises(ValueError, match="unsupported modifier_selection fields"):
        parse(unpenalized_interactions=True)
    assert effects.selected_modifier_penalty(default, "binary") == UNPENALIZED
    assert effects.selected_modifier_penalty(off, "binary") is None
    assert effects.selected_modifier_penalty(balanced, "continuous") is None


@pytest.mark.parametrize("binary", [True, False])
def test_final_estimation_applies_the_policy_only_under_orthogonal_tests(tmp_path, binary):
    kw = final_inputs(binary)
    default = small_policy()
    # The fixture has 72 training rows; one event per parameter lets the guard pass.
    run = {
        name: analysis.estimate_outer_fold(
            **kw, **final_candidates(kw), estimator="linear_interactions",
            statistical_policy=policy, output_dir=tmp_path / name,
        )
        for name, policy in (
            ("orthogonal", _with_selection(default, unpenalized_events_per_parameter=1.0)),
            ("disabled", _with_selection(default, unpenalized_selected_interactions=False)),
            ("balanced", _with_selection(default, rule="balanced_evidence")),
        )
    }
    predictions = {
        name: pd.read_csv(tmp_path / name / "predictions.csv") for name in run
    }
    summary = run["orthogonal"]["selected_modifier_penalty"]
    assert summary["enabled"] and summary["path"] == "unpenalized"
    assert summary["unpenalized_groups"] == ["main:x", "effect:x"]
    assert summary["attempted_paths"] == ["unpenalized"]
    record = run["orthogonal"]["interaction_fit_audit"]["selected_modifier_penalty"]
    assert record["path"] == "unpenalized"
    if binary:
        # The outcome is a noisy threshold of x: its main effect spans more than
        # the range limit, which is reported, while only interactions are checked.
        attempt = record["attempts"][0]
        assert attempt["max_main_effect_range_effect"] > attempt["max_range_effect_limit"]
        assert attempt["max_range_effect"] <= attempt["max_range_effect_limit"]
        assert attempt["max_range_effect_column"] == "treatment:x:value"
    assert "interaction_model_selected_modifier_penalty" in run["orthogonal"]["methodology_notes"]
    assert list((tmp_path / "orthogonal").glob(
        "fit_checkpoints/*/interaction_cv_selected_unpenalized/fold_*.json"))
    assert run["disabled"]["selected_modifier_penalty"] == {
        "enabled": False, "path": "fully_penalized", "reason": "disabled_by_configuration"}
    # balanced_evidence keeps its diagnostics and the single tuned penalty.
    assert "selected_modifier_penalty" not in run["balanced"]
    assert "selected_modifier_penalty" not in run["balanced"]["interaction_fit_audit"]
    assert "interaction_model_selected_modifier_penalty" not in run["balanced"]["methodology_notes"]
    pd.testing.assert_series_equal(
        predictions["balanced"].estimated_cate, predictions["disabled"].estimated_cate)
    assert not np.allclose(
        predictions["orthogonal"].estimated_cate.dropna(),
        predictions["disabled"].estimated_cate.dropna(),
    )
    # The setting is part of the estimation identity: a change refits.
    first = json.loads((tmp_path / "orthogonal" / "complete.json").read_text())
    analysis.estimate_outer_fold(
        **kw, **final_candidates(kw), estimator="linear_interactions",
        statistical_policy=_with_selection(default, unpenalized_events_per_parameter=2.0),
        output_dir=tmp_path / "orthogonal",
    )
    second = json.loads((tmp_path / "orthogonal" / "complete.json").read_text())
    assert first["input_fingerprint"] != second["input_fingerprint"]


def test_ensemble_component_audit_exposes_the_summary(tmp_path):
    kw = final_inputs(True)
    result = analysis.estimate_outer_fold(
        **kw, **final_candidates(kw), estimator="ensemble",
        statistical_policy=small_policy(), output_dir=tmp_path,
    )
    component = result["ensemble"]["component_audits"]["linear_interactions"]
    summary = component["selected_modifier_penalty"]
    guard = component["interaction_fit_audit"]["selected_modifier_penalty"]["guard"]
    # 72 rows, one modifier column: 2 parameters within min(events, non-events) / 10.
    assert summary["enabled"] is True and summary["guard_passed"] is guard["passed"] is True
    assert summary["path"] == "unpenalized"
    assert summary["unpenalized_parameters"] == 2
    assert summary["parameter_limit"] == pytest.approx(guard["information"] / 10.0)
    assert component["interaction_fit_audit"]["selected_modifier_penalty"]["path"] == (
        summary["path"])


def test_setting_changes_reuse_selection_and_preflight(tmp_path, monkeypatch):
    from oci.inference.plain_handoff_stage2 import (
        PlainHandoffStage2Config,
        Stage2ExtractionLLMConfig,
    )
    from oci.inference.research_all_evidence_workflow import (
        _stage2_reselection_policy_fingerprint,
    )
    from oci.inference.stage2_preflight import validate_selection_resume
    from tests.stage2_prompt_spy import install
    from tests.test_stage2_orthogonal_selection import _fold_inputs, _request_json

    install(monkeypatch)

    def run(output, policy):
        dataset, definitions, split = _fold_inputs()
        config = PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1", model="test-model",
            extraction_llm=Stage2ExtractionLLMConfig(
                endpoint="http://small-stage2.test/v1", model="small-test-model", workers=1),
            max_prompt_chars=12_000, extraction_max_prompt_chars=20_000, max_review_rounds=1,
            estimation_trees=12, nuisance_learner="elastic_net", min_informative_patients=2,
            min_propensity=0.05, max_propensity=0.95, statistical_selection=policy,
        )
        return config, analysis.run_fold_analysis(
            dataset=dataset, definitions=definitions, split=split,
            clinical_question="Estimate treatment effect.", unit_id_column="patient_id",
            text_column="clinical_text", treatment_column="treatment_indicator",
            outcome_column="outcome_indicator", outcome_type="binary", inner_folds=2, seed=11,
            output_dir=output, request_json=_request_json, config=config,
        )

    output = tmp_path / "stage2" / "outer_001"
    first_config, first = run(output, small_policy())
    selection_files = {
        path: path.read_bytes() for path in (output / "selection").rglob("*") if path.is_file()
    }

    def forbidden(**kwargs):
        raise AssertionError("selection reran after a final-estimation setting change")

    monkeypatch.setattr(count, "select_modifier_count", forbidden)
    monkeypatch.setattr(analysis, "select_stage2_features_multi_model", forbidden)
    changed = _with_selection(small_policy(), unpenalized_selected_interactions=False)
    second_config, second = run(output, changed)
    assert {path: path.read_bytes() for path in selection_files} == selection_files
    audits = [
        result["estimation"]["ensemble"]["component_audits"]["linear_interactions"]
        ["selected_modifier_penalty"]
        for result in (first, second)
    ]
    assert audits[0]["enabled"] is True and audits[1]["enabled"] is False
    # Preflight and the reselection identity also ignore the setting.
    root = tmp_path / "stage2"
    (root / "config.json").write_text(json.dumps({
        "statistical_selection": first_config.statistical_selection.public_dict(),
        "min_propensity": 0.05, "max_propensity": 0.95, "estimation_trees": 12,
        "propensity_clip": first_config.propensity_clip,
    }))
    validate_selection_resume(SimpleNamespace(stage2=second_config, output_dir=tmp_path))
    stage = SimpleNamespace(
        stage2=first_config, clinical_question="q", outcome_type="binary", outer_folds=2,
        inner_folds=2, seed=1)
    before = _stage2_reselection_policy_fingerprint(stage)
    after = _stage2_reselection_policy_fingerprint(
        SimpleNamespace(**{**vars(stage), "stage2": second_config}))
    assert before == after
    modifier = ModifierSelectionConfig()
    assert modifier.selection_dict() == replace(
        modifier, unpenalized_events_per_parameter=3.0).selection_dict()
