from __future__ import annotations


import pytest

from oci.inference import plain_handoff_stage2_analysis as analysis


def feature(name="brain_sites", **kwargs):
    return {
        "feature_id": "f_" + name, "name": name,
        "description": "Anatomical sites of brain metastases.",
        "value_type": "categorical", "categories_or_unit": ["frontal", "cerebellum"],
        "measurement_definition": "List all involved anatomical regions.",
        "missing_value_rule": "Return null when undocumented.",
        "supporting_packet_ids": ["card_1"], **kwargs,
    }


def split_answer():
    return {"action": "split", "reason": "Several anatomical sites can be involved simultaneously.",
            "components": [{
                "name": name, "description": f"Metastasis in the {name}.",
                "value_type": "binary", "categories_or_unit": ["Present", "Absent"],
                "label_definitions": {
                    "Present": {"meaning": f"Metastatic disease involves the {name}.",
                                "established_by": f"A documented metastasis in the {name}."},
                    "Absent": {"meaning": f"The {name} is free of metastatic disease.",
                               "established_by": f"An explicit statement of no metastasis in the {name}."},
                },
                "measurement_definition": f"Use the latest explicit finding of metastasis in the {name}. "
                    "Present requires positive evidence; Absent requires an explicit negative finding.",
                "missing_value_rule": "Return null if this site is not assessed.",
            } for name in ["frontal", "cerebellum"]]}


def test_split_is_stable_atomic_and_does_not_invent_patient_values():
    original = feature()
    decision = analysis._validate_ontology_refinement(split_answer(), feature=original)
    # Cached normalized decisions must round-trip without changing component IDs.
    assert analysis._validate_ontology_refinement(decision, feature=original) == decision
    children = analysis._apply_ontology_decision(original, decision)
    assert [c["name"] for c in children] == ["brain_sites__frontal", "brain_sites__cerebellum"]
    assert all(c["supporting_packet_ids"] == ["card_1"] for c in children)
    assert all(c["ontology_parent_feature_id"] == original["feature_id"] for c in children)
    assert all(c["conflict_resolution"]["strategy"] == "latest" for c in children)
    assert all("null when this component is not mentioned" in c["missing_value_rule"]
               for c in children)
    assert all("documented_unknown when the record states that it was not assessed"
               in c["missing_value_rule"] for c in children)
    assert all("explicit negative finding" in c["missing_value_rule"] for c in children)
    assert "brain_sites" not in [c["name"] for c in children]
    values = {c["clinical_label"]: "Present" for c in children}
    validated = analysis._validate_extraction(values, row_ids=[0], definitions=children)
    assert set(validated["rows"][0]["values"].values()) == {"Present"}
    with pytest.raises(analysis._ExtractionCategoryError):
        analysis._validate_extraction({"brain sites": "frontal, cerebellum"},
                                      row_ids=[0], definitions=[original])


@pytest.mark.parametrize("change", ["locked"])
def test_invalid_splits_and_locked_features_are_rejected(change):
    original, answer = feature(), split_answer()
    if change == "duplicate":
        answer["components"][1]["name"] = answer["components"][0]["name"]
    elif change == "one":
        answer["components"] = answer["components"][:1]
    else:
        original["configured_explicit_feature"] = True
    with pytest.raises(ValueError):
        analysis._validate_ontology_refinement(answer, feature=original)


def test_revisions_need_closed_types_and_label_definitions():
    original = feature()
    revision = {"action": "revise", "reason": "Clarify.", "definition": {
        "description": "Any brain metastasis.", "value_type": "binary",
        "categories_or_unit": ["Present", "Absent"],
        "label_definitions": split_answer()["components"][0]["label_definitions"],
        "measurement_definition": "Use the latest explicit finding.",
        "missing_value_rule": "Return null when undocumented."}}
    decision = analysis._validate_ontology_refinement(revision, feature=original)
    assert list(decision["label_definitions"]) == ["Present", "Absent"]
    assert analysis._validate_ontology_refinement(decision, feature=original) == decision
    [revised] = analysis._apply_ontology_decision(original, decision)
    assert revised["label_definitions"] == decision["label_definitions"]
    without = {**revision, "definition": {k: v for k, v in revision["definition"].items()
                                          if k != "label_definitions"}}
    with pytest.raises(ValueError, match="label_definitions"):
        analysis._validate_ontology_refinement(without, feature=original)
    free_text = {**revision, "definition": {**revision["definition"], "value_type": "ambiguous",
                                            "categories_or_unit": [], "label_definitions": {}}}
    with pytest.raises(ValueError, match="no free-text or ambiguous type"):
        analysis._validate_ontology_refinement(free_text, feature=original)


def test_ordinal_maximum_selects_the_highest_documented_grade():
    definition = {
        "feature_id": "f_toxicity", "name": "toxicity_grade", "description": "Worst toxicity grade.",
        "value_type": "ordinal", "categories_or_unit": ["None", "Mild", "Severe"],
        "measurement_definition": "Record the worst documented grade.",
        "missing_value_rule": "Null when undocumented.",
        "conflict_resolution": {"strategy": "maximum", "positive_category": None},
    }
    policy = analysis._resolved_conflict_resolution(definition)
    assert policy["strategy"] == "maximum"
    observations = [
        {"observation_id": "a", "value": "Severe", "page_index": 0, "observation_index": 0},
        {"observation_id": "b", "value": "mild", "page_index": 0, "observation_index": 1},
    ]
    value, decision = analysis._resolve_feature_observations(definition=definition, observations=observations)
    assert value == "Severe"
    assert decision["selection_basis"].startswith("maximum_ordinal_rank")
    lowest, _ = analysis._resolve_feature_observations(
        definition={**definition, "conflict_resolution": {"strategy": "minimum"}}, observations=observations)
    assert lowest == "mild"
    from oci.inference import stage2_clinical_prompts as clinical_prompts
    assert "highest documented category" in clinical_prompts.feature_text(definition)
    with pytest.raises(ValueError, match="continuous or ordinal"):
        analysis._resolved_conflict_resolution({**definition, "value_type": "categorical"})
