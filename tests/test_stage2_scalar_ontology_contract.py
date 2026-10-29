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
