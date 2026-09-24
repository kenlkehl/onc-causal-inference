from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from oci.inference.stage2_sequential_consolidation import (
    SCHEMA_VERSION,
    Stage2SequentialConsolidationConfig,
    consolidate_stage2_candidates,
    latent_states_for_selected,
    materialize_selected_latents,
    measurement_definitions_for_selected,
)


def _continuous(feature_id: str, name: str) -> dict[str, object]:
    return {
        "feature_id": feature_id,
        "name": name,
        "display_name": name.replace("_", " ").title(),
        "description": f"Pretreatment {name} measurement.",
        "value_type": "continuous",
        "modeling_strategy": "continuous",
        "categories_or_unit": ["unitless"],
        "measurement_definition": f"Use the recorded pretreatment {name}.",
        "missing_value_rule": "Null when absent.",
        "roles": [],
    }


def _categorical(feature_id: str, name: str) -> dict[str, object]:
    return {
        "feature_id": feature_id,
        "name": name,
        "display_name": name.replace("_", " ").title(),
        "description": "Pretreatment tumor histology.",
        "value_type": "categorical",
        "modeling_strategy": "categorical",
        "categories_or_unit": ["adenocarcinoma", "squamous"],
        "measurement_definition": "Use the recorded pretreatment histology.",
        "missing_value_rule": "Null when absent.",
        "roles": [],
    }


def _binary(feature_id: str, name: str) -> dict[str, object]:
    return {
        "feature_id": feature_id,
        "name": name,
        "display_name": name.replace("_", " ").title(),
        "description": f"Pretreatment {name} measurement.",
        "value_type": "binary",
        "modeling_strategy": "categorical",
        "categories_or_unit": ["Present", "Absent"],
        "measurement_definition": f"Use the recorded pretreatment {name}.",
        "missing_value_rule": "Null when absent.",
        "roles": [],
    }


def _ordinal(feature_id: str, name: str) -> dict[str, object]:
    return {
        "feature_id": feature_id,
        "name": name,
        "display_name": name.replace("_", " ").title(),
        "description": f"Pretreatment {name} ordinal measurement.",
        "value_type": "ordinal",
        "modeling_strategy": "categorical",
        "categories_or_unit": ["0", "1", "2", "3"],
        "measurement_definition": f"Use the recorded pretreatment {name} grade.",
        "missing_value_rule": "Null when absent.",
        "roles": [],
    }


def _embedding(texts, _model, _device):
    rows = []
    for text in texts:
        normalized = text.lower()
        if "third" in normalized:
            rows.append([0.0, 1.0])
        elif "histology" in normalized:
            rows.append([1.0, 0.0])
        else:
            rows.append([1.0, 0.0])
    return np.asarray(rows, dtype=float)


def _continuous_alias_response(source_ids):
    return {
        "action": "replace_with_latents",
        "rationale": "The two fields are alternate encodings of the same measurement.",
        "latents": [
            {
                "kind": "categorical_rule",
                "source_feature_ids": list(source_ids),
                "label": "Canonical baseline burden",
                "description": "The first documented value from equivalent burden fields.",
                "rationale": "Their definitions, units, granularity, and values align.",
                "measurement_definition": "Coalesce the equivalent source measurements.",
                "missing_value_rule": "Null only when every source is missing.",
                "output_type": "continuous",
                "categories_or_unit": ["unitless"],
                "expression": {
                    "op": "coalesce",
                    "feature_ids": list(source_ids),
                },
            }
        ],
    }


def test_created_latent_replaces_components_in_later_retrieval(tmp_path, monkeypatch):
    definitions = [
        _continuous("a", "first_burden"),
        _continuous("b", "second_burden"),
        _continuous("c", "third_marker"),
    ]
    frame = pd.DataFrame(
        {
            "_oci_row_id": np.arange(20),
            "first_burden": np.arange(20, dtype=float),
            "second_burden": np.arange(20, dtype=float),
            "third_marker": np.linspace(1.0, 2.0, 20),
        }
    )
    requests = []
    from oci.inference import stage2_sequential_consolidation as module
    original = module._decision_messages
    def capture(step_input, **kwargs):
        requests.append(step_input)
        return original(step_input, **kwargs)
    monkeypatch.setattr(module, "_decision_messages", capture)

    def request_json(messages, validate, *, request_kind="interpretation", **repair_kwargs):
        assert request_kind == "interpretation"
        assert "feature_id" not in messages[1]["content"]
        assert "Paired measurements" in messages[1]["content"]
        assert repair_kwargs["conservative_validation_fallback"]["action"] == "leave_unchanged"
        assert repair_kwargs["fallback_after_same_error"] == 3
        if len(requests) == 1:
            return validate({"action": "merge", "reason": "Equivalent values and units.",
                "members": ["First Burden", "Second Burden"],
                "canonical_label": "Baseline burden", "category_equivalences": []})
        return validate({"action": "keep", "reason": "Distinct clinical measurements."})

    consolidated, active, report, entries = consolidate_stage2_candidates(
        extracted_fit=frame,
        definitions=definitions,
        request_json=request_json,
        policy=Stage2SequentialConsolidationConfig(
            enabled=True,
            neighbor_count=1,
            embedding_model="test-model",
            embedding_device="cpu",
        ),
        output_dir=tmp_path / "consolidation",
        request_model="test-llm",
        embedding_function=_embedding,
    )

    assert report["schema_version"] == SCHEMA_VERSION
    assert report["latents_created"] == 1
    assert report["components_consumed"] == 2
    assert len(entries) == 1
    latent_id = entries[0]["latent_id"]
    assert [feature["feature_id"] for feature in active] == ["c", latent_id]
    assert entries[0]["name"] in consolidated
    assert len(requests) == 2
    second_cluster_ids = {
        feature["feature_id"] for feature in requests[1]["features"]
    }
    assert second_cluster_ids == {"c", latent_id}
    assert "a" not in second_cluster_ids
    assert "b" not in second_cluster_ids

    def unexpected_request(*_args, **_kwargs):
        raise AssertionError("completed consolidation should be reconstructed from checkpoint")

    resumed, resumed_active, resumed_report, resumed_entries = (
        consolidate_stage2_candidates(
            extracted_fit=frame,
            definitions=definitions,
            request_json=unexpected_request,
            policy=Stage2SequentialConsolidationConfig(
                enabled=True,
                neighbor_count=1,
                embedding_model="test-model",
                embedding_device="cpu",
            ),
            output_dir=tmp_path / "consolidation",
            request_model="test-llm",
            embedding_function=_embedding,
        )
    )
    assert resumed_report == report
    assert resumed_entries == entries
    assert [feature["feature_id"] for feature in resumed_active] == ["c", latent_id]
    np.testing.assert_allclose(
        resumed[entries[0]["name"]],
        consolidated[entries[0]["name"]],
        equal_nan=True,
    )


def test_categorical_latent_flattens_lineage_and_populates_heldout(tmp_path):
    definitions = [
        _categorical("h1", "histology_primary"),
        _categorical("h2", "histology_secondary"),
    ]
    frame = pd.DataFrame(
        {
            "_oci_row_id": np.arange(24),
            "histology_primary": [
                "adenocarcinoma",
                "squamous",
                None,
                None,
            ] * 6,
            "histology_secondary": [
                "adenocarcinoma",
                "squamous",
                "adenocarcinoma",
                "squamous",
            ] * 6,
        }
    )

    def request_json(
        _messages,
        validate,
        *,
        request_kind="interpretation",
        **_repair_kwargs,
    ):
        assert request_kind == "interpretation"
        return validate(
            {
                "action": "replace_with_latents",
                "rationale": "These are alternate fields for the same histology concept.",
                "latents": [
                    {
                        "kind": "categorical_rule",
                        "source_feature_ids": ["h1", "h2"],
                        "label": "Baseline tumor histology",
                        "description": "The first available pretreatment histology value.",
                        "rationale": "The fields are semantic and empirical aliases.",
                        "measurement_definition": (
                            "Use histology_primary when present, otherwise histology_secondary."
                        ),
                        "missing_value_rule": "Null when both source fields are absent.",
                        "output_type": "categorical",
                        "categories_or_unit": ["adenocarcinoma", "squamous"],
                        "expression": {
                            "op": "coalesce",
                            "feature_ids": ["h1", "h2"],
                        },
                    }
                ],
            }
        )

    consolidated, active, _report, entries = consolidate_stage2_candidates(
        extracted_fit=frame,
        definitions=definitions,
        request_json=request_json,
        policy=Stage2SequentialConsolidationConfig(
            enabled=True,
            neighbor_count=1,
            embedding_model="test-model",
        ),
        output_dir=tmp_path / "consolidation",
        request_model="test-llm",
        embedding_function=_embedding,
    )
    assert len(active) == 1
    latent = active[0]
    assert latent["derived_structured_latent"] is True
    assert latent["measurement_dependency_feature_ids"] == ["h1", "h2"]
    assert consolidated[latent["name"]].tolist() == [
        "adenocarcinoma",
        "squamous",
        "adenocarcinoma",
        "squamous",
    ] * 6

    selected = [{**latent, "roles": ["confounder"]}]
    dependencies = measurement_definitions_for_selected(selected, definitions)
    states = latent_states_for_selected(selected, entries)
    assert [feature["feature_id"] for feature in dependencies] == ["h1", "h2"]
    assert [item["latent_id"] for item in states] == [latent["feature_id"]]

    heldout = pd.DataFrame(
        {
            "_oci_row_id": [20, 21, 22],
            "histology_primary": [None, "squamous", None],
            "histology_secondary": ["adenocarcinoma", None, None],
        }
    )
    populated = materialize_selected_latents(
        frame=heldout,
        latent_states=states,
        measurement_definitions=dependencies,
    )
    assert populated[latent["name"]].tolist()[:2] == [
        "adenocarcinoma",
        "squamous",
    ]
    assert pd.isna(populated[latent["name"]].iloc[2])


def test_continuous_alias_preserves_nonnumeric_source_information(tmp_path):
    definitions = [
        _continuous("a", "first_measure"),
        _continuous("b", "second_measure"),
    ]
    frame = pd.DataFrame(
        {
            "_oci_row_id": np.arange(20),
            "first_measure": ["100/60 unitless", *map(float, range(101, 120))],
            "second_measure": np.arange(100, 120, dtype=float),
        }
    )

    def request_json(_messages, validate, **_kwargs):
        with pytest.raises(ValueError, match="discard reported thresholds"):
            validate(_continuous_alias_response(["a", "b"]))
        return validate({"action": "keep", "reason": "A source contains information that numeric coalescing would drop."})

    consolidated, active, report, entries = consolidate_stage2_candidates(
        extracted_fit=frame,
        definitions=definitions,
        request_json=request_json,
        policy=Stage2SequentialConsolidationConfig(
            enabled=True,
            neighbor_count=1,
            embedding_model="test-model",
        ),
        output_dir=tmp_path / "consolidation",
        request_model="test-llm",
        embedding_function=_embedding,
    )

    assert report["latents_created"] == 0
    assert len(active) == 2
    assert entries == []
    assert consolidated["first_measure"].iloc[0] == "100/60 unitless"


def test_continuous_alias_rejects_conflicting_overlapping_values(tmp_path):
    definitions = [
        _continuous("a", "first_measure"),
        _continuous("b", "second_measure"),
    ]
    first = np.arange(20, dtype=float)
    second = first.copy()
    second[7] = 7.25
    frame = pd.DataFrame(
        {
            "_oci_row_id": np.arange(100, 120),
            "first_measure": first,
            "second_measure": second,
        }
    )
    checked = False

    def request_json(_messages, validate, **_kwargs):
        nonlocal checked
        if not checked:
            checked = True
            with pytest.raises(
                ValueError,
                match="conflicting overlapping values.*conflict_count=1",
            ):
                validate(_continuous_alias_response(["a", "b"]))
        return validate(
            {
                "action": "leave_unchanged",
                "rationale": "The overlapping measurements disagree.",
                "latents": [],
            }
        )

    _consolidated, active, report, entries = consolidate_stage2_candidates(
        extracted_fit=frame,
        definitions=definitions,
        request_json=request_json,
        policy=Stage2SequentialConsolidationConfig(
            enabled=True,
            neighbor_count=1,
            embedding_model="test-model",
        ),
        output_dir=tmp_path / "consolidation",
        request_model="test-llm",
        embedding_function=_embedding,
    )

    assert checked is True
    assert len(active) == 2
    assert report["latents_created"] == 0
    assert entries == []


def test_categorical_coalesce_rejects_conflicting_overlapping_values(tmp_path):
    definitions = [
        _categorical("h1", "histology_primary"),
        _categorical("h2", "histology_secondary"),
    ]
    primary = ["adenocarcinoma", "squamous"] * 10
    secondary = list(primary)
    secondary[6] = "squamous"
    frame = pd.DataFrame(
        {
            "_oci_row_id": np.arange(20),
            "histology_primary": primary,
            "histology_secondary": secondary,
        }
    )
    checked = False

    def request_json(_messages, validate, **_kwargs):
        nonlocal checked
        if not checked:
            checked = True
            proposal = {
                "action": "replace_with_latents",
                "rationale": "Attempted alias coalesce.",
                "latents": [
                    {
                        "kind": "categorical_rule",
                        "source_feature_ids": ["h1", "h2"],
                        "label": "Canonical histology",
                        "description": "Canonical pretreatment histology.",
                        "rationale": "The fields appear to be aliases.",
                        "measurement_definition": "Coalesce the source fields.",
                        "missing_value_rule": "Null when both fields are absent.",
                        "output_type": "categorical",
                        "categories_or_unit": ["adenocarcinoma", "squamous"],
                        "expression": {
                            "op": "coalesce",
                            "feature_ids": ["h1", "h2"],
                        },
                    }
                ],
            }
            with pytest.raises(ValueError, match="conflicting overlapping values"):
                validate(proposal)
        return validate(
            {
                "action": "leave_unchanged",
                "rationale": "The overlapping category values disagree.",
                "latents": [],
            }
        )

    consolidate_stage2_candidates(
        extracted_fit=frame,
        definitions=definitions,
        request_json=request_json,
        policy=Stage2SequentialConsolidationConfig(
            enabled=True,
            neighbor_count=1,
            embedding_model="test-model",
        ),
        output_dir=tmp_path / "consolidation",
        request_model="test-llm",
        embedding_function=_embedding,
    )
    assert checked is True


def test_categorical_alias_recode_allows_lossless_union_vocabulary(tmp_path):
    definitions = [
        {
            **_categorical("h1", "cancer_histology"),
            "categories_or_unit": [
                "Adenocarcinoma",
                "Squamous cell carcinoma",
                "Large-cell carcinoma",
                "Other",
            ],
        },
        {
            **_categorical("h2", "lung_cancer_histology"),
            "categories_or_unit": [
                "Adenocarcinoma",
                "Squamous Cell Carcinoma",
                "Large Cell Carcinoma",
            ],
        },
    ]
    first_cycle = [
        "Adenocarcinoma",
        "Squamous cell carcinoma",
        "Large-cell carcinoma",
        "Other",
    ]
    second_cycle = [
        "Adenocarcinoma",
        "Squamous Cell Carcinoma",
        "Large Cell Carcinoma",
        None,
    ]
    frame = pd.DataFrame(
        {
            "_oci_row_id": np.arange(24),
            "cancer_histology": first_cycle * 6,
            "lung_cancer_histology": second_cycle * 6,
        }
    )

    def request_json(_messages, validate, **_kwargs):
        return validate(
            {
                "action": "replace_with_latents",
                "rationale": "The fields encode the same primary cancer histology.",
                "latents": [
                    {
                        "kind": "categorical_rule",
                        "source_feature_ids": ["h1", "h2"],
                        "label": "Primary cancer histology",
                        "description": "Canonical primary cancer histology.",
                        "rationale": (
                            "The narrower vocabulary is a lossless subset of the "
                            "canonical union."
                        ),
                        "measurement_definition": (
                            "Map synonymous histology spellings to canonical labels."
                        ),
                        "missing_value_rule": "Null when both sources are missing.",
                        "output_type": "categorical",
                        "categories_or_unit": [
                            "Adenocarcinoma",
                            "Squamous cell carcinoma",
                            "Large cell carcinoma",
                            "Other",
                        ],
                        "expression": {
                            "op": "case",
                            "cases": [
                                {
                                    "when": {
                                        "feature_id": "h1",
                                        "operator": "eq",
                                        "value": source,
                                    },
                                    "then": target,
                                }
                                for source, target in [
                                    ("Adenocarcinoma", "Adenocarcinoma"),
                                    (
                                        "Squamous cell carcinoma",
                                        "Squamous cell carcinoma",
                                    ),
                                    ("Large-cell carcinoma", "Large cell carcinoma"),
                                    ("Other", "Other"),
                                ]
                            ]
                            + [
                                {
                                    "when": {
                                        "feature_id": "h2",
                                        "operator": "eq",
                                        "value": source,
                                    },
                                    "then": target,
                                }
                                for source, target in [
                                    ("Adenocarcinoma", "Adenocarcinoma"),
                                    (
                                        "Squamous Cell Carcinoma",
                                        "Squamous cell carcinoma",
                                    ),
                                    ("Large Cell Carcinoma", "Large cell carcinoma"),
                                ]
                            ],
                            "else": None,
                        },
                    }
                ],
            }
        )

    consolidated, active, report, entries = consolidate_stage2_candidates(
        extracted_fit=frame,
        definitions=definitions,
        request_json=request_json,
        policy=Stage2SequentialConsolidationConfig(
            enabled=True,
            neighbor_count=1,
            embedding_model="test-model",
        ),
        output_dir=tmp_path / "consolidation",
        request_model="test-llm",
        embedding_function=_embedding,
    )

    assert report["latents_created"] == 1
    assert len(active) == 1
    assert len(entries) == 1
    assert consolidated[entries[0]["name"]].tolist() == [
        "Adenocarcinoma",
        "Squamous cell carcinoma",
        "Large cell carcinoma",
        "Other",
    ] * 6


def test_categorical_recode_rejects_conflicting_canonical_values(tmp_path):
    definitions = [
        _binary("a", "first_status"),
        {
            **_binary("b", "second_status"),
            "categories_or_unit": ["Yes", "No"],
        },
    ]
    first = ["Present", "Absent"] * 10
    second = ["Yes", "No"] * 10
    second[4] = "No"
    frame = pd.DataFrame(
        {
            "_oci_row_id": np.arange(20),
            "first_status": first,
            "second_status": second,
        }
    )
    checked = False

    def request_json(_messages, validate, **_kwargs):
        nonlocal checked
        if not checked:
            checked = True
            proposal = {
                "action": "replace_with_latents",
                "rationale": "Attempted synonymous recode.",
                "latents": [
                    {
                        "kind": "categorical_rule",
                        "source_feature_ids": ["a", "b"],
                        "label": "Canonical status",
                        "description": "Canonical pretreatment status.",
                        "rationale": "The source vocabularies are synonymous.",
                        "measurement_definition": "Map both vocabularies.",
                        "missing_value_rule": "Null when both sources are absent.",
                        "output_type": "binary",
                        "categories_or_unit": ["Present", "Absent"],
                        "expression": {
                            "op": "case",
                            "cases": [
                                {
                                    "when": {
                                        "feature_id": "a",
                                        "operator": "eq",
                                        "value": source,
                                    },
                                    "then": target,
                                }
                                for source, target in (
                                    ("Present", "Present"),
                                    ("Absent", "Absent"),
                                )
                            ]
                            + [
                                {
                                    "when": {
                                        "feature_id": "b",
                                        "operator": "eq",
                                        "value": source,
                                    },
                                    "then": target,
                                }
                                for source, target in (
                                    ("Yes", "Present"),
                                    ("No", "Absent"),
                                )
                            ],
                            "else": None,
                        },
                    }
                ],
            }
            with pytest.raises(ValueError, match="conflicting overlapping values"):
                validate(proposal)
        return validate(
            {
                "action": "leave_unchanged",
                "rationale": "The canonicalized overlapping values disagree.",
                "latents": [],
            }
        )

    consolidate_stage2_candidates(
        extracted_fit=frame,
        definitions=definitions,
        request_json=request_json,
        policy=Stage2SequentialConsolidationConfig(
            enabled=True,
            neighbor_count=1,
            embedding_model="test-model",
        ),
        output_dir=tmp_path / "consolidation",
        request_model="test-llm",
        embedding_function=_embedding,
    )
    assert checked is True


def test_categorical_union_recode_rejects_within_source_category_collapse(tmp_path):
    definitions = [
        {
            **_categorical("h1", "first_histology"),
            "categories_or_unit": ["Adenocarcinoma", "Squamous", "Other"],
        },
        {
            **_categorical("h2", "second_histology"),
            "categories_or_unit": ["Adenocarcinoma", "Squamous", "Other"],
        },
    ]
    values = ["Adenocarcinoma", "Squamous", "Other"] * 7
    frame = pd.DataFrame(
        {
            "_oci_row_id": np.arange(21),
            "first_histology": values,
            "second_histology": values,
        }
    )
    checked = False

    def request_json(_messages, validate, **_kwargs):
        nonlocal checked
        if not checked:
            checked = True
            cases = []
            for feature_id in ("h1", "h2"):
                for category in ("Adenocarcinoma", "Squamous", "Other"):
                    cases.append(
                        {
                            "when": {
                                "feature_id": feature_id,
                                "operator": "eq",
                                "value": category,
                            },
                            "then": (
                                "Carcinoma"
                                if category in {"Adenocarcinoma", "Squamous"}
                                else "Other"
                            ),
                        }
                    )
            proposal = {
                "action": "replace_with_latents",
                "rationale": "Attempted lossy rollup.",
                "latents": [
                    {
                        "kind": "categorical_rule",
                        "source_feature_ids": ["h1", "h2"],
                        "label": "Collapsed histology",
                        "description": "A lossy histology rollup.",
                        "rationale": "Collapse distinct histologies.",
                        "measurement_definition": "Collapse source categories.",
                        "missing_value_rule": "Null when both are missing.",
                        "output_type": "categorical",
                        "categories_or_unit": ["Carcinoma", "Other"],
                        "expression": {"op": "case", "cases": cases, "else": None},
                    }
                ],
            }
            with pytest.raises(ValueError, match="collapses distinct categories"):
                validate(proposal)
        return validate(
            {
                "action": "leave_unchanged",
                "rationale": "The proposed recode loses category information.",
                "latents": [],
            }
        )

    consolidate_stage2_candidates(
        extracted_fit=frame,
        definitions=definitions,
        request_json=request_json,
        policy=Stage2SequentialConsolidationConfig(
            enabled=True,
            neighbor_count=1,
            embedding_model="test-model",
        ),
        output_dir=tmp_path / "consolidation",
        request_model="test-llm",
        embedding_function=_embedding,
    )
    assert checked is True


def test_rejects_alias_proposal_below_pairwise_association_threshold(tmp_path):
    definitions = [_continuous("a", "first_measure"), _continuous("b", "second_measure")]
    frame = pd.DataFrame(
        {
            "_oci_row_id": np.arange(20),
            "first_measure": np.arange(20, dtype=float),
            "second_measure": np.tile([0.0, 1.0], 10),
        }
    )
    checked = False

    def request_json(_messages, validate, **_kwargs):
        nonlocal checked
        if not checked:
            checked = True
            with pytest.raises(ValueError, match="minimum_pairwise_association=0.850"):
                validate(_continuous_alias_response(["a", "b"]))
        return validate(
            {
                "action": "leave_unchanged",
                "rationale": "The fields do not meet the empirical alias threshold.",
                "latents": [],
            }
        )

    _consolidated, active, report, entries = consolidate_stage2_candidates(
        extracted_fit=frame,
        definitions=definitions,
        request_json=request_json,
        policy=Stage2SequentialConsolidationConfig(
            enabled=True,
            neighbor_count=1,
            embedding_model="test-model",
        ),
        output_dir=tmp_path / "consolidation",
        request_model="test-llm",
        embedding_function=_embedding,
    )
    assert checked is True
    assert len(active) == 2
    assert report["latents_created"] == 0
    assert entries == []


def test_rejects_information_losing_ordinal_to_binary_replacement(tmp_path):
    definitions = [
        _ordinal("grade", "functional_grade"),
        _binary("status", "functional_status"),
    ]
    grades = ["0", "1", "2", "3"] * 5
    frame = pd.DataFrame(
        {
            "_oci_row_id": np.arange(20),
            "functional_grade": grades,
            "functional_status": [
                "Present" if value in {"0", "1"} else "Absent" for value in grades
            ],
        }
    )
    checked = False

    def request_json(_messages, validate, **_kwargs):
        nonlocal checked
        if not checked:
            checked = True
            proposal = {
                "action": "replace_with_latents",
                "rationale": "The binary field is derived from the ordinal field.",
                "latents": [
                    {
                        "kind": "categorical_rule",
                        "source_feature_ids": ["grade", "status"],
                        "label": "Functional status",
                        "description": "A coarsened functional indicator.",
                        "rationale": "The fields are strongly related.",
                        "measurement_definition": "Map both fields to a binary status.",
                        "missing_value_rule": "Null when both fields are absent.",
                        "output_type": "binary",
                        "categories_or_unit": ["Present", "Absent"],
                        "expression": {
                            "op": "case",
                            "cases": [
                                {
                                    "when": {
                                        "feature_id": "grade",
                                        "operator": "in",
                                        "values": ["0", "1"],
                                    },
                                    "then": "Present",
                                },
                                {
                                    "when": {
                                        "feature_id": "grade",
                                        "operator": "in",
                                        "values": ["2", "3"],
                                    },
                                    "then": "Absent",
                                },
                                {
                                    "when": {
                                        "feature_id": "status",
                                        "operator": "eq",
                                        "value": "Present",
                                    },
                                    "then": "Present",
                                },
                                {
                                    "when": {
                                        "feature_id": "status",
                                        "operator": "eq",
                                        "value": "Absent",
                                    },
                                    "then": "Absent",
                                },
                            ],
                            "else": None,
                        },
                    }
                ],
            }
            with pytest.raises(ValueError, match="same value_type"):
                validate(proposal)
        return validate(
            {
                "action": "leave_unchanged",
                "rationale": "A coarser representation must not replace the ordinal source.",
                "latents": [],
            }
        )

    consolidate_stage2_candidates(
        extracted_fit=frame,
        definitions=definitions,
        request_json=request_json,
        policy=Stage2SequentialConsolidationConfig(
            enabled=True,
            neighbor_count=1,
            embedding_model="test-model",
        ),
        output_dir=tmp_path / "consolidation",
        request_model="test-llm",
        embedding_function=_embedding,
    )
    assert checked is True


def test_rejects_case_rule_that_turns_missingness_into_absence(tmp_path):
    definitions = [_binary("a", "first_status"), _binary("b", "second_status")]
    values = ["Present", "Absent"] * 10
    frame = pd.DataFrame(
        {
            "_oci_row_id": np.arange(20),
            "first_status": values,
            "second_status": values,
        }
    )
    checked = False

    def request_json(_messages, validate, **_kwargs):
        nonlocal checked
        if not checked:
            checked = True
            cases = []
            for feature_id in ("a", "b"):
                for category in ("Present", "Absent"):
                    cases.append(
                        {
                            "when": {
                                "feature_id": feature_id,
                                "operator": "eq",
                                "value": category,
                            },
                            "then": category,
                        }
                    )
            proposal = {
                "action": "replace_with_latents",
                "rationale": "The fields are exact aliases.",
                "latents": [
                    {
                        "kind": "categorical_rule",
                        "source_feature_ids": ["a", "b"],
                        "label": "Canonical status",
                        "description": "Canonical representation of the status.",
                        "rationale": "The fields have the same meaning and encoding.",
                        "measurement_definition": "Map synonymous status labels.",
                        "missing_value_rule": "Null when both fields are absent.",
                        "output_type": "binary",
                        "categories_or_unit": ["Present", "Absent"],
                        "expression": {
                            "op": "case",
                            "cases": cases,
                            "else": "Absent",
                        },
                    }
                ],
            }
            with pytest.raises(ValueError, match="else=null"):
                validate(proposal)
        return validate(
            {
                "action": "leave_unchanged",
                "rationale": "The proposed rule did not preserve missingness.",
                "latents": [],
            }
        )

    consolidate_stage2_candidates(
        extracted_fit=frame,
        definitions=definitions,
        request_json=request_json,
        policy=Stage2SequentialConsolidationConfig(
            enabled=True,
            neighbor_count=1,
            embedding_model="test-model",
        ),
        output_dir=tmp_path / "consolidation",
        request_model="test-llm",
        embedding_function=_embedding,
    )
    assert checked is True


def test_disabled_consolidation_is_identity(tmp_path):
    definitions = [_continuous("a", "first"), _continuous("b", "second")]
    frame = pd.DataFrame(
        {"_oci_row_id": [0, 1], "first": [1.0, 2.0], "second": [3.0, 4.0]}
    )

    def unexpected_request(*_args, **_kwargs):
        raise AssertionError("disabled consolidation must not call the LLM")

    consolidated, active, report, entries = consolidate_stage2_candidates(
        extracted_fit=frame,
        definitions=definitions,
        request_json=unexpected_request,
        policy=Stage2SequentialConsolidationConfig(enabled=False),
        output_dir=tmp_path / "consolidation",
        request_model="test-llm",
        embedding_function=_embedding,
    )
    pd.testing.assert_frame_equal(consolidated, frame)
    assert active == definitions
    assert report["status"] == "disabled"
    assert entries == []
    persisted_report = json.loads(
        (tmp_path / "consolidation" / "report.json").read_text(encoding="utf-8")
    )
    assert persisted_report == report


def test_single_candidate_consolidation_writes_advertised_report(tmp_path):
    definitions = [_continuous("a", "only_measure")]
    frame = pd.DataFrame({"_oci_row_id": [0, 1], "only_measure": [1.0, 2.0]})

    def unexpected_request(*_args, **_kwargs):
        raise AssertionError("a single candidate must not call the LLM")

    _consolidated, active, report, entries = consolidate_stage2_candidates(
        extracted_fit=frame,
        definitions=definitions,
        request_json=unexpected_request,
        policy=Stage2SequentialConsolidationConfig(enabled=True),
        output_dir=tmp_path / "consolidation",
        request_model="test-llm",
        embedding_function=_embedding,
    )

    assert active == definitions
    assert report["status"] == "complete_fewer_than_two_candidates"
    assert entries == []
    assert json.loads(
        (tmp_path / "consolidation" / "report.json").read_text(encoding="utf-8")
    ) == report


def test_invalid_response_and_conservative_fallback_are_persisted(tmp_path):
    definitions = [_continuous("a", "first"), _continuous("b", "second")]
    frame = pd.DataFrame(
        {
            "_oci_row_id": np.arange(12),
            "first": np.arange(12, dtype=float),
            "second": np.arange(12, dtype=float) + 0.1,
        }
    )
    request_count = 0

    def request_json(
        _messages,
        validate,
        *,
        request_kind="interpretation",
        validation_event_observer=None,
        conservative_validation_fallback=None,
        **_repair_kwargs,
    ):
        nonlocal request_count
        request_count += 1
        assert request_kind == "interpretation"
        if request_count == 1:
            assert validation_event_observer is not None
            assert conservative_validation_fallback is not None
            validation_event_observer(
                {
                    "event": "invalid_response",
                    "response_attempt": 1,
                    "validation_failure_count": 1,
                    "same_error_occurrence": 1,
                    "error_type": "ValueError",
                    "error_message": "categorical_rule.expression must be an object",
                    "raw_response": '{"action":"replace_with_latents"}',
                    "parsed_response": {"action": "replace_with_latents"},
                }
            )
            fallback = validate(conservative_validation_fallback)
            validation_event_observer(
                {
                    "event": "conservative_fallback",
                    "response_attempt": 1,
                    "validation_failure_count": 1,
                    "same_error_occurrence": 1,
                    "trigger": "validation_repairs_exhausted",
                    "error_type": "ValueError",
                    "error_message": "categorical_rule.expression must be an object",
                    "fallback_response": fallback,
                }
            )
            return fallback
        return validate(
            {
                "action": "leave_unchanged",
                "rationale": "The measurements remain distinct.",
                "latents": [],
            }
        )

    _consolidated, active, report, entries = consolidate_stage2_candidates(
        extracted_fit=frame,
        definitions=definitions,
        request_json=request_json,
        policy=Stage2SequentialConsolidationConfig(
            enabled=True,
            neighbor_count=1,
            embedding_model="test-model",
        ),
        output_dir=tmp_path / "consolidation",
        request_model="test-llm",
        embedding_function=_embedding,
    )

    assert len(active) == 2
    assert report["latents_created"] == 0
    assert entries == []
    repair_dir = (
        tmp_path
        / "consolidation"
        / "steps"
        / "step_0001_a"
        / "repair_attempts"
    )
    attempt = json.loads((repair_dir / "attempt_0001.json").read_text())
    fallback = json.loads((repair_dir / "fallback.json").read_text())
    assert attempt["raw_response"] == '{"action":"replace_with_latents"}'
    assert attempt["error_message"] == "categorical_rule.expression must be an object"
    assert fallback["trigger"] == "validation_repairs_exhausted"
    decision = json.loads(
        (
            tmp_path
            / "consolidation"
            / "steps"
            / "step_0001_a"
            / "decision.json"
        ).read_text()
    )
    assert decision["action"] == "leave_unchanged"
    assert "Conservative schema-validation fallback" in decision["rationale"]
