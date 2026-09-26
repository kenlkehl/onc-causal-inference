"""Real numerical comparisons, training boundaries, and safe checkpoint reuse."""

from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from oci.inference import stage2_estimand_ontology as search
from oci.inference.stage2_estimand_ontology_config import (
    EstimandOntologyConfig, estimand_ontology_config_from_mapping,
)
from oci.inference.plain_handoff_stage2 import plain_stage2_config_from_mapping


def feature(name):
    return {"name": name, "feature_id": f"private_{name}", "description": f"Baseline {name}.",
            "value_type": "continuous", "categories_or_unit": [],
            "measurement_definition": "Extract the documented pretreatment value.",
            "missing_value_rule": "Use null for missing or unresolved findings.",
            "roles": ["confounder", "effect_modifier"], "ground_truth": "DO_NOT_SHOW"}


def experiment(tmp_path, *, binary=False):
    rng = np.random.default_rng(825)
    n = 480
    c, m = rng.normal(size=(2, n))
    t = rng.binomial(1, 1 / (1 + np.exp(-0.6 * c)))
    y = 2 * c + t * (0.5 + 3 * m) + rng.normal(scale=0.4, size=n)
    if binary:
        y = rng.binomial(1, 1 / (1 + np.exp(-y)))
    ids = np.arange(n) * 3 + 20
    base = pd.DataFrame({"_oci_row_id": ids, "c": c + rng.normal(scale=3, size=n),
                         "m": m + rng.normal(scale=3, size=n)})
    labels = pd.DataFrame({"_oci_row_id": ids, "treatment": t, "outcome": y})
    splits = [{"fit_row_ids": ids[a].tolist(), "heldout_row_ids": ids[b].tolist()}
              for a, b in StratifiedKFold(3, shuffle=True, random_state=8).split(base, t)]
    calls, extractions = [], []

    def request(messages, validate, **kwargs):
        text = json.dumps(messages)
        assert "DO_NOT_SHOW" not in text and "private_" not in text
        assert "screening_scores" not in text and "r_loss" not in text
        assert not kwargs
        calls.append(messages)
        return validate({"reason": "Preserve an explicitly documented pretreatment measurement.",
            "alternatives": [{"rationale": "Use the numerical pretreatment measurement.",
                "description": "Explicit pretreatment numerical measurement.", "value_type": "continuous",
                "categories_or_unit": [], "measurement_definition": "Extract the last pretreatment measured value.",
                "missing_value_rule": "Use null when unavailable."}]})

    def extract(definitions, directory):
        extractions.append(deepcopy(definitions))
        values = pd.DataFrame({"_oci_row_id": ids})
        for f in definitions:
            values[f["name"]] = c if f["estimand_ontology"]["source_feature_name"] == "c" else m
        return values, definitions

    return dict(extracted_fit=base, labels=labels, definitions=[feature("c"), feature("m")],
                inner_splits=splits, outcome_type="binary" if binary else "continuous",
                clinical_question="Compare treatment A versus B for this pretreatment cohort.",
                request_json=request, extract_alternatives=extract, output_dir=tmp_path,
                policy=EstimandOntologyConfig(enabled=True, max_features_per_role=2, ridge_penalty=2),
                source_text_fingerprint="training-text-only", request_identity={"model": "test-model"}, seed=88), calls, extractions


@pytest.mark.parametrize("binary", [False, True])
def test_real_search_compares_both_roles_and_reuses_completed_work(tmp_path, binary):
    arguments, calls, extractions = experiment(tmp_path, binary=binary)
    frame, definitions, report = search.refine_estimand_ontologies(**arguments)
    assert len(calls) == 2 and len(extractions) == 1 and len(extractions[0]) == 2
    assert len(report["accepted_feature_ids"]) >= 1
    pd.testing.assert_frame_equal(frame[arguments["extracted_fit"].columns], arguments["extracted_fit"])
    assert definitions[:2] == arguments["definitions"]
    accepted_roles = {role for f in definitions[2:] for role in f["estimand_ontology"]["supported_uses"]}
    if not binary:
        assert accepted_roles == {"confounder", "effect_modifier"}
    assert report["original_measurements_retained"] and not report["outer_test_used"]
    for path in tmp_path.glob("runs/*/reference_*.json"):
        ref = json.loads(path.read_text())["result"]
        assert set(ref["fit_row_ids"]).isdisjoint(ref["validation_row_ids"])
        for inner in ref["crossfit_audit"]:
            assert set(inner["fit_row_ids"]).isdisjoint(inner["validation_row_ids"])
            assert set(inner["fit_row_ids"]) | set(inner["validation_row_ids"]) == set(ref["fit_row_ids"])
    for feature_dir in tmp_path.glob("runs/*/features/*"):
        baseline = json.loads((feature_dir / "baseline_scores.json").read_text())["result"]
        for path in feature_dir.glob("scores_*.json"):
            scores = json.loads(path.read_text())["result"]
            assert [s["overlap_validation_row_ids"] for s in scores] == [s["overlap_validation_row_ids"] for s in baseline]
    arguments.update(request_json=lambda *a, **k: pytest.fail("proposal repeated"),
                     extract_alternatives=lambda *a, **k: pytest.fail("extraction repeated"))
    resumed = search.refine_estimand_ontologies(**arguments)
    pd.testing.assert_frame_equal(resumed[0], frame)
    assert resumed[1:] == (definitions, report)
    # A saved augmented preselection matrix also resumes from the original catalog.
    augmented = search.refine_estimand_ontologies(**{**arguments, "extracted_fit": frame, "definitions": definitions})
    pd.testing.assert_frame_equal(augmented[0], frame)
    assert augmented[1:] == (definitions, report)


def test_confounded_and_instrument_only_alternatives_are_not_equivalent_evidence():
    policy = EstimandOntologyConfig()
    base = [{"outcome_loss": 1., "propensity_loss": 1., "overlap_fraction": .9, "r_loss": 1.}] * 3
    def comparison(key, q, e, overlap, r):
        return {"feature_id": key, "baseline": base, "alternative": [
            {"outcome_loss": q, "propensity_loss": e, "overlap_fraction": overlap, "r_loss": r}] * 3}
    decision = search.choose_alternatives([
        comparison("instrument", 1., .5, .9, 1.),
        comparison("bad_overlap", .5, .5, .2, 1.),
        comparison("adjustment", .8, .9, .9, 1.),
        comparison("modifier", 1., 1., .9, .7),
    ], policy)
    assert decision["winners"] == {"confounder": "adjustment", "effect_modifier": "modifier"}
    assert not search.paired_gain([1, 1, 1], [.8, 1.1, 1.1], policy)["accepted"]


def test_alternatives_are_bounded_and_remove_stale_representation_state():
    original = {**feature("functional_status"), "modeling_strategy": "categorical",
                "harmonization_plan": {"old": "mapping"}, "conflict_resolution": {"strategy": "maximum"}}
    alternative = {k: original[k] for k in search.DEFINITION_FIELDS}
    alternative.update(description="Explicit ECOG score.", rationale="Use the named clinical scale.",
                       value_type="ordinal", categories_or_unit=["0", "1", "2", "3", "4"])
    validated = search.validate_proposals({"reason": "A named scale is available.", "alternatives": [alternative]},
                                         feature=original, maximum=2)
    variant = search._variant(original, validated["alternatives"][0])
    assert "harmonization_plan" not in variant and "modeling_strategy" not in variant
    assert variant["conflict_resolution"]["strategy"] != "maximum"
    assert "__estimand_" not in search.prompts.feature_text(variant)
    from oci.inference.plain_handoff_stage2_analysis import _extraction_prompt, _validate_extraction
    message = _extraction_prompt(definitions=[variant], rows=[{"text": "ECOG 1"}])
    assert "__estimand_" not in json.dumps(message)
    measured = _validate_extraction({search.prompts.label(variant): "1"}, row_ids=[2], definitions=[variant])
    assert measured["rows"][0]["values"][variant["name"]] == "1"
    # Both downstream prompt projections preserve the same human label that
    # their validators resolve back to the Python-owned measurement identity.
    from oci.inference.stage2_role_adjudication import _prompt_safe_definition, Stage2RoleAdjudicationConfig
    from oci.inference.stage2_sequential_consolidation import _prompt_feature
    safe = _prompt_safe_definition(variant, policy=Stage2RoleAdjudicationConfig())
    matrix = pd.DataFrame({variant["name"]: ["0", "1", "2"]})
    projected = _prompt_feature(variant, frame=matrix, cosine_similarity=1, protected=False)
    for view in (safe, projected):
        assert search.prompts.label(view) == search.prompts.label(variant)
        assert "__estimand_" not in search.prompts.feature_text(view)
    for proposals in ([alternative] * 3, [alternative] * 2, [{**alternative, "feature_id": "model_should_not_supply"}]):
        with pytest.raises(ValueError):
            search.validate_proposals({"reason": "Clinical rationale", "alternatives": proposals}, feature=original, maximum=2)
    with pytest.raises(ValueError, match="cannot be revised"):
        search.validate_proposals({"reason": "Clinical rationale", "alternatives": [alternative]},
                                  feature={**original, "configured_explicit_feature": True}, maximum=2)


def test_search_boundaries_and_insufficient_data_preserve_originals(tmp_path):
    arguments, calls, extractions = experiment(tmp_path)
    malformed = deepcopy(arguments["inner_splits"])
    malformed[0]["heldout_row_ids"].append(999999)
    with pytest.raises(ValueError, match="exactly the training"):
        search.refine_estimand_ontologies(**{**arguments, "inner_splits": malformed})
    with pytest.raises(ValueError, match="only training"):
        search.refine_estimand_ontologies(**{**arguments, "labels": arguments["labels"].assign(oracle=123)})
    arguments["policy"] = replace(arguments["policy"], minimum_training_rows=1000)
    frame, definitions, report = search.refine_estimand_ontologies(**arguments)
    assert report["status"] == "not_estimable" and not calls and not extractions
    pd.testing.assert_frame_equal(frame, arguments["extracted_fit"])
    assert definitions == arguments["definitions"]


def test_protected_features_no_proposals_and_input_changes_invalidate_search(tmp_path):
    arguments, calls, extractions = experiment(tmp_path)
    arguments["definitions"][0]["configured_explicit_feature"] = True
    search.refine_estimand_ontologies(**arguments)
    assert len(calls) == 1 and all(f["estimand_ontology"]["source_feature_name"] == "m" for batch in extractions for f in batch)
    arguments["labels"] = arguments["labels"].copy()
    arguments["labels"].loc[0, "outcome"] += .1
    search.refine_estimand_ontologies(**arguments)
    assert len(calls) == 2 and len(list(tmp_path.glob("runs/*/result.json"))) == 2
    path = next(tmp_path.glob("runs/*/result.json"))
    value = json.loads(path.read_text())
    value["result"]["accepted_definitions"].append({"name": "tampered"})
    path.write_text(json.dumps(value))
    # Corruption is always rejected for the matching input directory.
    with pytest.raises(ValueError, match="corrupt"):
        search._checkpoint(path.parent, "result", value["input_fingerprint"], lambda: pytest.fail("corrupt cache recomputed"))


def test_configuration_is_opt_in_and_roundtrips():
    config = plain_stage2_config_from_mapping({"endpoint": "http://test/v1"}, default_workers=1)
    assert not config.estimand_ontology.enabled
    config = plain_stage2_config_from_mapping({"endpoint": "http://test/v1", "estimand_ontology": {
        "enabled": True, "max_features_per_role": 5}}, default_workers=1)
    assert plain_stage2_config_from_mapping(config.public_dict(), default_workers=1).estimand_ontology == config.estimand_ontology
    for value in ({"enabled": "yes"}, {"max_features_per_role": True}, {"unknown": 1},
                  {"ridge_penalty": 0}, {"minimum_relative_gain": float("nan")}):
        with pytest.raises(ValueError):
            estimand_ontology_config_from_mapping(value)


def test_extraction_outage_resumes_from_proposals_without_repeating_requests(tmp_path):
    arguments, calls, _ = experiment(tmp_path)
    extract = arguments["extract_alternatives"]
    def unavailable(*args):
        raise RuntimeError("extractor offline")
    with pytest.raises(RuntimeError, match="extractor offline"):
        search.refine_estimand_ontologies(**{**arguments, "extract_alternatives": unavailable})
    assert len(calls) == 2 and not list(tmp_path.glob("runs/*/result.json"))
    result = search.refine_estimand_ontologies(**{**arguments, "extract_alternatives": extract,
        "request_json": lambda *a, **k: pytest.fail("completed proposal repeated after outage")})
    assert result[2]["status"] == "complete"


def test_guarded_reselection_uses_versioned_matrix_and_rejects_path_escape(tmp_path):
    from tests.test_research_all_evidence_workflow import _completed_stage2_reselection_fixture
    from oci.inference import research_all_evidence_workflow as workflow

    config = _completed_stage2_reselection_fixture(tmp_path)
    root = config.output_dir / "stage2"
    old_paths = []
    for outer in sorted(root.glob("outer_*")):
        relative = "extraction/estimand_candidates_fit/version/extracted.csv"
        original = outer / "extraction/all_candidates_fit/extracted.csv"
        changed = outer / relative
        changed.parent.mkdir(parents=True)
        changed.write_bytes(original.read_bytes())
        old_paths.append((original, original.read_bytes()))
        path = outer / "selection/input.json"
        value = json.loads(path.read_text())
        value["preselection_matrix_path"] = relative
        value.pop("input_fingerprint")
        digest = workflow._stage2_value_fingerprint(value)
        path.write_text(json.dumps({**value, "input_fingerprint": digest}))
        complete_path = outer / "selection/complete.json"
        complete = json.loads(complete_path.read_text())
        complete["input_fingerprint"] = digest
        complete_path.write_text(json.dumps(complete))
    # Path rejection occurs before mutation, even with a matching input hash.
    path = root / "outer_001/selection/input.json"
    original_input = json.loads(path.read_text())
    value = {**original_input, "preselection_matrix_path": "../outside.csv"}
    value.pop("input_fingerprint")
    digest = workflow._stage2_value_fingerprint(value)
    path.write_text(json.dumps({**value, "input_fingerprint": digest}))
    marker = path.parent / "complete.json"
    completion = json.loads(marker.read_text())
    marker.write_text(json.dumps({**completion, "input_fingerprint": digest}))
    with pytest.raises(RuntimeError, match="under the fold extraction"):
        workflow.prepare_stage2_reselection(config=config)
    path.write_text(json.dumps(original_input))
    marker.write_text(json.dumps(completion))
    workflow.prepare_stage2_reselection(config=config)
    for outer in root.glob("outer_*"):
        snapshot = json.loads((outer / "preselection/input.json").read_text())
        assert snapshot["matrix_path"] == "extraction/estimand_candidates_fit/version/extracted.csv"
        assert (outer / snapshot["matrix_path"]).exists()
    assert all(path.read_bytes() == content for path, content in old_paths)


def test_pipeline_freezes_winners_before_heldout_extraction_and_passes_to_selection(tmp_path, monkeypatch):
    from oci.inference import plain_handoff_stage2_analysis as analysis
    from oci.inference.plain_handoff_stage2 import PlainHandoffStage2Config
    from oci.inference.stage2_role_adjudication import Stage2RoleAdjudicationConfig

    definition = feature("c")
    dataset = pd.DataFrame({"id": range(40), "text": ["baseline measurement"] * 40,
                            "t": [0, 1] * 20, "y": [0, 1, 1, 0] * 10,
                            "oracle": ["DO_NOT_READ"] * 40})
    train_ids, test_ids = list(range(24)), list(range(24, 40))
    events = []

    def initial(**kw):
        assert kw["row_ids"] == train_ids
        path = kw["feedback_dir"] / "final_failure_summary.json"
        analysis._write_json(path, {})
        return pd.DataFrame({"_oci_row_id": train_ids, "c": np.arange(24)}), kw["definitions"], 0

    def measure(**kw):
        events.append(("extract", kw["row_ids"], [f["name"] for f in kw["definitions"]]))
        if kw["row_ids"] == train_ids:
            assert list(kw["dataset"].columns) == ["text"]
            assert all(f.get("estimand_ontology") for f in kw["definitions"])
        else:
            assert kw["row_ids"] == test_ids and any(event[0] == "selected" for event in events)
        return pd.DataFrame({"_oci_row_id": kw["row_ids"], **{
            f["name"]: np.arange(len(kw["row_ids"])) for f in kw["definitions"]}})

    def refine(**kw):
        assert kw["labels"]._oci_row_id.tolist() == train_ids
        assert list(kw["labels"].columns) == ["_oci_row_id", "treatment", "outcome"]
        assert kw["extracted_fit"]._oci_row_id.tolist() == train_ids
        alternative = {k: definition[k] for k in search.DEFINITION_FIELDS}
        alternative.update(description="Precisely timed baseline C.", rationale="Resolve the pretreatment time.")
        variant = search._variant(definition, alternative)
        measurements, variants = kw["extract_alternatives"]([variant], kw["output_dir"] / "test")
        return kw["extracted_fit"].merge(measurements, on="_oci_row_id"), [definition, *variants], {
            "status": "complete", "input_fingerprint": "accepted-definition-search"}

    def select(arguments):
        assert len(arguments["definitions"]) == 2
        assert arguments["extracted_fit"]._oci_row_id.tolist() == train_ids
        selected = [{**f, "roles": ["confounder", "effect_modifier"]} for f in arguments["definitions"]]
        events.append(("selected", selected))
        return selected, {"schema_version": analysis.STAGE2_ROLE_SELECTION_SCHEMA_VERSION}, selected, []

    def estimate(**kw):
        assert len(kw["definitions"]) == 2
        assert kw["extracted_heldout"]._oci_row_id.tolist() == test_ids
        assert (tmp_path / "final_definitions.json").is_file()
        events.append(("estimated",))
        return {"status": "test_estimation"}

    monkeypatch.setattr(analysis, "_extract_training_with_ontology_feedback", initial)
    monkeypatch.setattr(analysis, "_harmonize_training_extraction", lambda **kw: (kw["extracted"], kw["definitions"], {}))
    monkeypatch.setattr(analysis, "_request_aggregate_ontology_supervisor", lambda **kw: (kw["definitions"], False, {"changed_feature_ids": []}))
    monkeypatch.setattr(analysis, "extract_rows", measure)
    monkeypatch.setattr(analysis, "_run_stage2_statistical_selection", select)
    monkeypatch.setattr(analysis, "estimate_outer_fold", estimate)
    monkeypatch.setattr(search, "refine_estimand_ontologies", refine)
    result = analysis.run_fold_analysis(
        dataset=dataset, definitions=[definition], split={"fit_row_ids": train_ids, "heldout_row_ids": test_ids},
        clinical_question="Compare A and B.", unit_id_column="id", text_column="text", treatment_column="t",
        outcome_column="y", outcome_type="binary", inner_folds=2, seed=1, output_dir=tmp_path,
        request_json=lambda *a, **k: pytest.fail("unexpected LLM request"),
        config=PlainHandoffStage2Config(endpoint="http://test/v1", model="test", max_review_rounds=1,
            estimand_ontology=EstimandOntologyConfig(enabled=True),
            role_adjudication=Stage2RoleAdjudicationConfig(enabled=False)),
    )
    assert result["estimand_ontology"]["status"] == "complete" and events[-1] == ("estimated",)
    saved = json.loads((tmp_path / "selection/input.json").read_text())
    assert saved["estimand_ontology"]["search_fingerprint"] == "accepted-definition-search"
    assert pd.read_csv(tmp_path / "extraction/all_candidates_fit/extracted.csv").shape[1] == 2
    assert pd.read_csv(tmp_path / saved["preselection_matrix_path"]).shape[1] == 3
