"""Real numerical comparisons, training boundaries, and safe checkpoint reuse."""

from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from oci.inference import stage2_estimand_ontology as search
from oci.inference.stage2_estimand_ontology_config import (
    EstimandOntologyConfig,
)


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


@pytest.mark.parametrize("binary", [False])
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


@pytest.mark.parametrize("estimand_enabled,sparse_alternative", [(True, False)])
def test_pipeline_freezes_winners_before_heldout_extraction_and_passes_to_selection(
    tmp_path, monkeypatch, estimand_enabled, sparse_alternative,
):
    from oci.inference import plain_handoff_stage2_analysis as analysis
    from oci.inference.plain_handoff_stage2 import PlainHandoffStage2Config
    from oci.inference.stage2_role_adjudication import Stage2RoleAdjudicationConfig

    definition = feature("c")
    dataset = pd.DataFrame({"id": range(40), "text": ["baseline measurement"] * 40,
                            "t": [0, 1] * 20, "y": [0, 1, 1, 0] * 10,
                            "oracle": ["DO_NOT_READ"] * 40})
    train_ids, test_ids = list(range(24)), list(range(24, 40))
    events = []
    selected_count = 1 + int(estimand_enabled and not sparse_alternative)

    def initial(**kw):
        assert kw["row_ids"] == train_ids
        path = kw["feedback_dir"] / "final_failure_summary.json"
        analysis._write_json(path, {})
        return pd.DataFrame({"_oci_row_id": train_ids, "c": np.arange(24),
                             "rare": [1] + [None] * 23}), kw["definitions"], 0

    def measure(**kw):
        events.append(("extract", kw["row_ids"], [f["name"] for f in kw["definitions"]]))
        assert all(f["name"] != "rare" for f in kw["definitions"])
        if kw["row_ids"] == train_ids:
            assert list(kw["dataset"].columns) == ["text"]
            assert all(f.get("estimand_ontology") for f in kw["definitions"])
        else:
            assert kw["row_ids"] == test_ids and any(event[0] == "selected" for event in events)
        return pd.DataFrame({"_oci_row_id": kw["row_ids"], **{
            f["name"]: ([1] + [None] * 23 if sparse_alternative and kw["row_ids"] == train_ids
                        else np.arange(len(kw["row_ids"]))) for f in kw["definitions"]}})

    def refine(**kw):
        assert estimand_enabled
        assert [f["name"] for f in kw["definitions"]] == ["c"]
        assert list(kw["extracted_fit"].columns) == ["_oci_row_id", "c"]
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
        assert len(arguments["definitions"]) == selected_count
        assert "rare" not in arguments["extracted_fit"]
        assert arguments["extracted_fit"]._oci_row_id.tolist() == train_ids
        selected = [{**f, "roles": ["confounder", "effect_modifier"]} for f in arguments["definitions"]]
        events.append(("selected", selected))
        return selected, {"schema_version": analysis.STAGE2_ROLE_SELECTION_SCHEMA_VERSION}, selected, []

    def estimate(**kw):
        assert len(kw["definitions"]) == selected_count
        assert kw["extracted_heldout"]._oci_row_id.tolist() == test_ids
        assert (tmp_path / "final_definitions.json").is_file()
        events.append(("estimated",))
        return {"status": "test_estimation"}

    monkeypatch.setattr(analysis, "_extract_training_with_ontology_feedback", initial)
    monkeypatch.setattr(analysis, "_harmonize_training_extraction", lambda **kw: (kw["extracted"], kw["definitions"], {}))
    monkeypatch.setattr(analysis, "_request_aggregate_ontology_supervisor", lambda **kw: (kw["definitions"], False, {"changed_feature_ids": []}))
    monkeypatch.setattr(analysis, "extract_rows", measure)
    monkeypatch.setattr(analysis, "_run_stage2_statistical_selection", select)
    from oci.inference import stage2_modifier_count
    monkeypatch.setattr(stage2_modifier_count, "select_modifier_count",
                        lambda **kw: (kw["selected"], kw["role_report"], None))
    monkeypatch.setattr(analysis, "estimate_outer_fold", estimate)
    monkeypatch.setattr(search, "refine_estimand_ontologies", refine)
    result = analysis.run_fold_analysis(
        dataset=dataset, definitions=[definition, feature("rare")],
        split={"fit_row_ids": train_ids, "heldout_row_ids": test_ids},
        clinical_question="Compare A and B.", unit_id_column="id", text_column="text", treatment_column="t",
        outcome_column="y", outcome_type="binary", inner_folds=2, seed=1, output_dir=tmp_path,
        request_json=lambda *a, **k: pytest.fail("unexpected LLM request"),
        config=PlainHandoffStage2Config(endpoint="http://test/v1", model="test", max_review_rounds=1,
            estimand_ontology=EstimandOntologyConfig(enabled=estimand_enabled),
            role_adjudication=Stage2RoleAdjudicationConfig()),
    )
    assert events[-1] == ("estimated",)
    assert result["candidate_missingness_filter"]["dropped_feature_ids"] == ["private_rare"]
    audit = json.loads((tmp_path / "extraction/candidate_missingness_filter.json").read_text())
    assert audit["rows"] == 24 and audit["features"][1]["observed_rows"] == 1
    saved = json.loads((tmp_path / "selection/input.json").read_text())
    if estimand_enabled:
        assert result["estimand_ontology"]["status"] == "complete"
        assert saved["estimand_ontology"]["search_fingerprint"] == "accepted-definition-search"
        assert result["candidate_missingness_filter"]["after_estimand_refinement"]["candidates_dropped"] == int(sparse_alternative)
    assert saved["candidate_missingness_filter"]["min_informative_patients"] == 20
    assert saved["candidate_missingness_filter"]["max_extraction_failure_fraction"] == 0.10
    assert audit["features"][1]["reason"] == "fewer_than_min_informative_patients_observed"
    assert pd.read_csv(tmp_path / "extraction/all_candidates_fit/extracted.csv").shape[1] == 3
    assert list(pd.read_csv(tmp_path / saved["preselection_matrix_path"]).columns) == [
        "_oci_row_id", *[f["name"] for f in saved["definitions"]]]
