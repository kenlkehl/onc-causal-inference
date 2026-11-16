"""Closed-format definitions and harmonized merging (stage2_variable_ontology)."""
from __future__ import annotations

import copy

import pytest

from oci.inference import stage2_clinical_prompts as clinical_prompts
from oci.inference import stage2_variable_ontology as ontology


def labelled(labels):
    return {label: {"meaning": f"The patient's state is {label}.",
                    "established_by": f"A documented {label}."} for label in labels}


def variable(name="finding_severity", *, value_type="ordinal", labels=("None", "Mild", "Severe"),
             unit=None, **changes):
    labels = list(labels) if value_type != "continuous" else []
    value = {
        "name": name, "description": "One attribute.", "value_type": value_type,
        "unit": unit, "categories": labels, "label_definitions": labelled(labels),
        "measurement_definition": "Use the latest documented value.",
        "missing_value_rule": "Use null when not mentioned.",
        "conflict_resolution": {"strategy": "latest", "positive_category": None},
    }
    value.update(changes)
    return value


def define(*variables, action="define"):
    return {"action": action, "reason": "Test.", "variables": list(variables)}


def problems(value):
    with pytest.raises(ValueError) as caught:
        ontology.validate_definition_response(value)
    return str(caught.value)


def test_valid_definitions_normalize_and_revalidate_unchanged():
    reply = define(variable(name="Finding Severity"))
    normalized = ontology.validate_definition_response(reply)
    assert normalized["variables"][0]["name"] == "finding_severity"
    assert ontology.validate_definition_response(normalized) == normalized
    decomposed = define(variable("first_finding", value_type="binary", labels=["Present", "Absent"]),
                        variable("second_quantity", value_type="continuous", unit="mg/dL"),
                        action="decompose")
    assert len(ontology.validate_definition_response(decomposed)["variables"]) == 2
    empty_rule = define(variable(missing_value_rule=""))
    assert "documented_unknown" in ontology.validate_definition_response(empty_rule)[
        "variables"][0]["missing_value_rule"]


@pytest.mark.parametrize("change, message", [
    ({"value_type": "ambiguous"}, "no free-text or ambiguous type"),
    ({"categories": ["None", "Mild", "Unknown"], "label_definitions": labelled(["None", "Mild", "Unknown"])},
     "unknown or undetermined state"),
    ({"categories": ["Positive", "Negative", "Not tested"],
      "label_definitions": labelled(["Positive", "Negative", "Not tested"]), "value_type": "categorical"},
     "not done"),
    ({"categories": ["M0", "M1", "M1a"], "label_definitions": labelled(["M0", "M1", "M1a"])}, "sublevel"),
    ({"categories": ["Stage III", "Stage IV", "Stage IVB"],
      "label_definitions": labelled(["Stage III", "Stage IV", "Stage IVB"])}, "sublevel"),
    ({"label_definitions": labelled(["None", "Mild"])}, "keys must be exactly its labels"),
    ({"label_definitions": {**labelled(["None", "Mild"]),
                            "Severe": {"meaning": "Documented as severe.", "established_by": "Severe."}}},
     "repeats how a record words the label"),
    ({"categories": ["0", "1-2", "3"], "label_definitions": labelled(["0", "1-2", "3"])}, "would be read as"),
    ({"conflict_resolution": {"strategy": "any_positive", "positive_category": "Mild"}}, "any_positive"),
])
def test_closed_format_problems_are_reported(change, message):
    assert message in problems(define(variable(**change)))


def test_not_done_labels_are_allowed_for_a_binary_performed_variable():
    performed = variable("test_performed", value_type="binary", labels=["Performed", "Not performed"])
    assert ontology.validate_definition_response(define(performed))


def test_score_with_few_levels_must_be_ordinal_but_wide_scores_stay_continuous():
    graded = variable("symptom_score", value_type="continuous", unit=None,
                      measurement_definition="Record the documented score from 0-4.")
    assert "ordinal labels for its levels" in problems(define(graded))
    wide = variable("functioning_score", value_type="continuous", unit=None,
                    measurement_definition="Record the 0-100 score converted from raw 1-4 responses.")
    assert ontology.validate_definition_response(define(wide))
    measured = variable("hemoglobin_concentration", value_type="continuous", unit="g/dL",
                        measurement_definition="Record the value with 1-2 decimals.")
    assert ontology.validate_definition_response(define(measured))
    assert "names more than one scale" in problems(define(variable(
        "symptom_score", value_type="continuous", unit="points on either scale")))


def test_decomposition_bounds_and_distinct_names():
    assert "2 to 12" in problems(define(variable(), action="decompose"))
    assert "distinct" in problems(define(variable(), variable(), action="decompose"))
    fallback = ontology.definition_fallback("no reply")
    assert ontology.validate_definition_response(fallback)["variables"] == []


def entry(entry_id, definition, *, locked=False):
    return {"entry_id": entry_id, "definition": definition, "locked": locked, "explicit_feature": None,
            "origin_variable_ids": [entry_id], "origin_candidate_ids": [f"candidate_{entry_id}"],
            "supporting_packet_ids": [f"packet_{entry_id}"], "supporting_architectures": ["test"],
            "label_map": {f"{entry_id}|{label}": label for label in definition["categories"]}}


def merge(members, definition, crosswalk):
    return {"groups": [{"members": members, "reason": "Same attribute.", "definition": definition,
                        "category_crosswalk": [{"member": m, "member_label": a, "shared_label": b}
                                               for m, a, b in crosswalk]}]}


def batch(*definitions, locked=()):
    return {f"V{index:02d}": entry(f"e{index}", copy.deepcopy(definition), locked=index in locked)
            for index, definition in enumerate(definitions, start=1)}


def merge_problems(value, members):
    with pytest.raises(ValueError) as caught:
        ontology.validate_merge_response(value, members=members)
    return str(caught.value)


def test_crosswalk_maps_every_label_and_composes_provenance():
    members = batch(variable(labels=["None", "Mild", "Severe"]),
                    variable("other_grade", labels=["absent", "mild", "severe"]))
    shared = variable(labels=["None", "Mild", "Severe"])
    reply = merge(["V01", "V02"], shared, [
        *(("V01", label, label) for label in ["None", "Mild", "Severe"]),
        ("V02", "Absent", "None"), ("V02", "mild", "Mild"), ("V02", "severe", "Severe"),
    ])
    normalized = ontology.validate_merge_response(reply, members=members)
    assert ontology.validate_merge_response(normalized, members=members) == normalized
    pool, summaries = ontology.apply_merge_groups(list(members.values()), list(members), normalized,
                                                  entry_prefix="r001_b001")
    [merged] = pool
    assert merged["label_map"]["e2|absent"] == "None"
    assert merged["origin_candidate_ids"] == ["candidate_e1", "candidate_e2"]
    assert summaries[0]["members"] == ["e1", "e2"]
    incomplete = merge(["V01", "V02"], shared, [("V01", label, label) for label in ["None", "Mild", "Severe"]])
    assert "must map every label of V02" in merge_problems(incomplete, members)


def test_merge_rejects_meaning_changes():
    staging = batch(variable("stage", labels=["Stage III", "Stage IV"]),
                    variable("stage_detail", labels=["Stage IVA", "Stage IVB"]))
    sublevel = merge(["V01", "V02"], variable("stage", labels=["Stage III", "Stage IVA", "Stage IVB"]), [
        ("V01", "Stage III", "Stage III"), ("V01", "Stage IV", "Stage IVB"),
        ("V02", "Stage IVA", "Stage IVA"), ("V02", "Stage IVB", "Stage IVB")])
    assert "to its sublevel" in merge_problems(sublevel, staging)

    systems = batch(variable("stage", labels=["I", "II"]), variable("extent", labels=["Local", "Distant"]))
    disjoint = merge(["V01", "V02"], variable("stage", labels=["I", "II", "Local", "Distant"]),
                     [("V01", "I", "I"), ("V01", "II", "II"), ("V02", "Local", "Local"),
                      ("V02", "Distant", "Distant")])
    assert "share no answer" in merge_problems(disjoint, systems)

    coarse = batch(variable("marker_result", value_type="binary", labels=["Positive", "Negative"]),
                   variable("marker_variant", value_type="categorical", labels=["Wild-type", "Variant A", "Other"]))
    specific_to_residual = merge(
        ["V01", "V02"], variable("marker_variant", value_type="categorical",
                                 labels=["Wild-type", "Variant A", "Other"]),
        [("V01", "Positive", "Other"), ("V01", "Negative", "Wild-type"), ("V02", "Wild-type", "Wild-type"),
         ("V02", "Variant A", "Variant A"), ("V02", "Other", "Other")])
    assert "specific label 'Positive'" in merge_problems(specific_to_residual, coarse)

    histology = batch(variable("tumour_type", value_type="categorical", labels=["Type A", "Not otherwise specified"]),
                      variable("tumour_type_b", value_type="categorical", labels=["Type A", "Organ X cancer NOS"]))
    narrower = merge(["V01", "V02"], variable("tumour_type", value_type="categorical",
                                              labels=["Type A", "Organ X cancer NOS"]),
                     [("V01", "Type A", "Type A"), ("V01", "Not otherwise specified", "Organ X cancer NOS"),
                      ("V02", "Type A", "Type A"), ("V02", "Organ X cancer NOS", "Organ X cancer NOS")])
    assert "residual of a narrower class" in merge_problems(narrower, histology)

    pair = batch(variable(labels=["None", "Mild"]), variable("other_grade", labels=["None", "Mild"]))
    invented = merge(["V01", "V02"], variable(labels=["None", "Mild", "Severe"]),
                     [(m, label, label) for m in ("V01", "V02") for label in ["None", "Mild"]])
    assert "no member label maps to (Severe)" in merge_problems(invented, pair)

    mixed = batch(variable(), variable("finding_value", value_type="continuous", unit="mm"))
    assert "mixes numbers and labels" in merge_problems(merge(["V01", "V02"], variable(), []), mixed)


def test_locked_members_keep_their_definition():
    fixed = variable("configured_grade", labels=["0", "1"], label_definitions={})
    members = batch(fixed, variable("found_grade", labels=["zero", "one"]), locked=(1,))
    reply = merge(["V01", "V02"], {"name": "configured_grade", "value_type": "ordinal", "unit": None,
                                   "categories": ["0", "1"]},
                  [("V01", "0", "0"), ("V01", "1", "1"), ("V02", "zero", "0"), ("V02", "one", "1")])
    normalized = ontology.validate_merge_response(reply, members=members)
    assert normalized["groups"][0]["definition"] == members["V01"]["definition"]
    renamed = copy.deepcopy(reply)
    renamed["groups"][0]["definition"]["name"] = "found_grade"
    assert "locked variable V01" in merge_problems(renamed, members)
    two_locked = batch(fixed, variable("configured_other", labels=["0", "1"], label_definitions={}),
                       locked=(1, 2))
    assert "more than one locked" in merge_problems(reply, two_locked)


def test_exact_duplicates_names_and_dispositions():
    first = entry("d0001_01", variable())
    second = entry("d0002_01", variable(categories=["none", "mild", "severe"],
                                        label_definitions=labelled(["none", "mild", "severe"])))
    third = entry("d0003_01", variable("finding_severity", value_type="binary", labels=["Present", "Absent"]))
    pool, merges = ontology.coalesce_exact_duplicates([first, second, third], entry_prefix="exact_r000")
    assert len(pool) == 2 and len(merges) == 1
    merged = next(item for item in pool if len(item["origin_candidate_ids"]) == 2)
    assert merged["label_map"]["d0002_01|mild"] == "Mild"
    names = ontology.unique_feature_names(pool)
    assert sorted(names.values()) == ["finding_severity", "finding_severity_2"]
    decomposed = {**entry("d0004_01", variable("part_a")), "origin_candidate_ids": ["candidate_x"]}
    other_part = {**entry("d0004_02", variable("part_b")), "origin_candidate_ids": ["candidate_x"]}
    everything = [*pool, decomposed, other_part]
    names = ontology.unique_feature_names(everything)
    result = ontology.dispositions(everything, names, {"candidate_y": "No reply."},
                                   ["candidate_d0001_01", "candidate_d0002_01", "candidate_x", "candidate_y"])
    assert result["candidate_d0001_01"]["status"] == "retained"
    assert result["candidate_d0002_01"]["status"] == "merged"
    assert result["candidate_x"]["status"] == "decomposed"
    assert result["candidate_y"] == {"status": "dropped", "feature_name": None, "feature_names": [],
                                     "reason": "No reply."}
    feature = ontology.feature_from_entry(merged, name="finding_severity")
    assert feature["categories_or_unit"] == ["None", "Mild", "Severe"]
    assert list(feature["label_definitions"]) == ["None", "Mild", "Severe"]


def test_prompts_render_label_definitions_without_identifiers():
    text, identifiers = ontology.merge_listing([entry("d0001_01", variable()),
                                                entry("c0002", variable("configured"), locked=True)])
    assert identifiers == ["V01", "V02"]
    assert "d0001_01" not in text and "candidate_" not in text and "packet_" not in text
    assert "meaning: The patient's state is Mild." in text and "Locked" in text
    feature = ontology.feature_from_entry(entry("d0001_01", variable()), name="finding_severity")
    rendered = clinical_prompts.feature_text(feature)
    assert "- Label definitions:" in rendered
    assert "  - Mild: The patient's state is Mild. Established by: A documented Mild." in rendered
    # Labels no longer declared are never shown.
    revised = {**feature, "categories_or_unit": ["None", "Severe"]}
    assert "Mild:" not in clinical_prompts.feature_text(revised)
    request = ontology.definition_request_text(name="finding_severity", descriptions=["One.", "Two."],
                                               rationales=["Seen in notes."], evidence=["Excerpt."])
    assert request.startswith("Candidate variable: finding severity\nDescription: One.")
    assert "Other descriptions of this candidate: Two." in request


def test_absent_level_labels_survive_measurement_csv_round_trips(tmp_path):
    import pandas as pd

    from oci.inference import stage2_diagnostics
    from oci.inference.plain_handoff_stage2_analysis import _write_frame

    path = tmp_path / "extracted.csv"
    _write_frame(path, pd.DataFrame({"_oci_row_id": [0, 1, 2],
                                     "finding_severity": ["None", None, "NA"]}))
    frame = stage2_diagnostics._read_measurements(path)
    assert frame["finding_severity"].tolist()[0] == "None"
    assert pd.isna(frame["finding_severity"].tolist()[1])
    assert frame["finding_severity"].tolist()[2] == "NA"


@pytest.mark.parametrize("change, message", [
    ({"categories": ["Stage I", "Stage II", "Unstaged"],
      "label_definitions": labelled(["Stage I", "Stage II", "Unstaged"])}, "unknown or undetermined state"),
    ({"categories": ["Low", "High", "Not graded"],
      "label_definitions": labelled(["Low", "High", "Not graded"])}, "not done"),
])
def test_unstaged_and_ungraded_labels_are_unknown_type(change, message):
    assert message in problems(define(variable(**change)))


def test_continuous_variable_must_use_one_scale_unless_a_conversion_is_stated():
    several = variable("symptom_score", value_type="continuous", unit="points",
                       measurement_definition="Record the number from validated patient-reported outcome "
                                              "instruments or clinical scales.")
    assert "several instruments or scales" in problems(define(several))
    converted = variable("symptom_score", value_type="continuous", unit="points",
                         measurement_definition="Record the number from validated instruments, converted to "
                                                "the 0-100 scale with the stated conversion.")
    assert ontology.validate_definition_response(define(converted))


def test_repeated_observation_rule_names_are_resolved_or_taken_from_members():
    assert ontology.canonical_strategy("latest observation") == "latest"
    assert ontology.canonical_strategy("Most recent documentation") == "latest"
    assert ontology.canonical_strategy("explicit_statement_precedence") is None
    members = batch(variable(labels=["None", "Mild"]), variable("other_grade", labels=["None", "Mild"]))
    shared = variable(labels=["None", "Mild"],
                      conflict_resolution={"strategy": "explicit statement precedence", "positive_category": None})
    reply = merge(["V01", "V02"], shared, [(m, label, label) for m in ("V01", "V02") for label in ["None", "Mild"]])
    normalized = ontology.validate_merge_response(reply, members=members)
    assert normalized["groups"][0]["definition"]["conflict_resolution"]["strategy"] == "latest"
    flags = batch(variable("flag", value_type="binary", labels=["Yes", "No"],
                           conflict_resolution={"strategy": "any_positive", "positive_category": "Yes"}),
                  variable("flag_b", value_type="binary", labels=["present", "absent"],
                           conflict_resolution={"strategy": "any_positive", "positive_category": "present"}))
    shared_flag = variable("flag", value_type="binary", labels=["Present", "Absent"],
                           conflict_resolution={"strategy": "precedence"})
    reply = merge(["V01", "V02"], shared_flag, [("V01", "Yes", "Present"), ("V01", "No", "Absent"),
                                                ("V02", "present", "Present"), ("V02", "absent", "Absent")])
    rule = ontology.validate_merge_response(reply, members=flags)["groups"][0]["definition"]["conflict_resolution"]
    assert rule == {"strategy": "any_positive", "positive_category": "Present"}
