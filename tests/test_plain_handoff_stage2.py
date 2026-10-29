from __future__ import annotations

from tests.stage2_prompt_spy import prompt_inputs

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import oci.inference.plain_handoff_stage2 as stage2_workflow
import oci.inference.plain_handoff_stage2_analysis as stage2_analysis

from oci.inference.plain_handoff_stage2 import (
    PlainHandoffStage2,
    PlainHandoffStage2Config,
    plain_stage2_config_from_mapping,
    run_plain_handoff_stage2,
)
from oci.inference.plain_handoff_stage2_analysis import extract_rows, run_fold_analysis


class _CharacterChatTokenizer:
    """Deterministic test tokenizer with one token per rendered character."""

    def __call__(
        self,
        text,
        *,
        add_special_tokens=False,
        return_attention_mask=False,
    ):
        del add_special_tokens, return_attention_mask
        return {"input_ids": list(range(len(str(text))))}

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=True,
        add_generation_prompt=True,
    ):
        assert tokenize is True
        rendered = "".join(
            f"<{message['role']}>{message['content']}" for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        return list(range(len(rendered)))


def _original_evidence_packets(packet_ids):
    return [
        {
            "packet_id": packet_id,
            "content": {
                "evidence_kind": "clinical_text",
                "representative_evidence": [
                    {"text": "Original clinical evidence for the candidate feature."}
                ],
            },
        }
        for packet_id in packet_ids
    ]


def _prompt_job(body):
    if "job" in body:
        return body["job"]
    if set(body) == {"candidate_feature_name", "supporting_evidence"}:
        return "operationalize_stage2_candidate_group"
    raise AssertionError(f"unrecognized prompt body keys: {sorted(body)}")


def _page_observation(
    *,
    feature_name,
    value,
    text,
    evidence,
    recorded_at=None,
    recorded_at_evidence=None,
):
    evidence_start = text.index(evidence)
    if recorded_at is None:
        recorded_at_start = None
        recorded_at_end = None
    else:
        recorded_at_start = text.index(recorded_at_evidence)
        recorded_at_end = recorded_at_start + len(recorded_at_evidence)
    return {
        "feature_name": feature_name,
        "value": value,
        "evidence": evidence,
        "evidence_start": evidence_start,
        "evidence_end": evidence_start + len(evidence),
        "recorded_at": recorded_at,
        "recorded_at_evidence": recorded_at_evidence,
        "recorded_at_start": recorded_at_start,
        "recorded_at_end": recorded_at_end,
    }


_INVALID_STAGE2_CONFIG_VALUES = [
    ("max_tokens", 99_999, "max_tokens must be an integer of at least 100000"),
]


@pytest.mark.parametrize(("field_name", "value", "message"), _INVALID_STAGE2_CONFIG_VALUES)
def test_stage2_config_rejects_invalid_values(field_name, value, message):
    with pytest.raises(ValueError, match=message):
        PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
            **{field_name: value},
        ).validate()


def test_stage2_explicit_feature_requires_complete_ontology():
    with pytest.raises(ValueError, match="requires ontology field"):
        plain_stage2_config_from_mapping(
            {
                "endpoint": "http://stage2.test/v1",
                "explicit_features": [
                    {
                        "name": "ecog_performance_status",
                        "roles": ["confounder"],
                    }
                ],
            },
            default_workers=1,
        )


_COMPLETE_EXPLICIT_FEATURE = {
    "name": "ecog", "roles": ["confounder"], "description": "ECOG status.",
    "value_type": "ordinal", "categories_or_unit": ["0-4"],
    "measurement_definition": "Latest pretreatment ECOG.",
    "missing_value_rule": "Null when undocumented.",
}


@pytest.mark.parametrize(
    "raw,section",
    [
        ({"endpont": "http://s.test/v1"}, "stage2 "),
        ({"endpoint": "http://s.test/v1",
          "extraction_llm": {"endpoint": "http://e.test/v1", "wokers": 2}},
         "stage2.extraction_llm "),
        ({"endpoint": "http://s.test/v1",
          "explicit_features": [{**_COMPLETE_EXPLICIT_FEATURE, "conflict_resoluton": "max"}]},
         r"stage2.explicit_features\[0\] "),
    ],
)
def test_stage2_unknown_keys_fail_closed_at_every_level(raw, section):
    with pytest.raises(ValueError, match=section) as caught:
        plain_stage2_config_from_mapping(raw, default_workers=1)
    assert re.search("unsupported|unknown", str(caught.value))


def test_stage2_resolved_snapshots_and_shipped_configs_still_parse(tmp_path):
    from dataclasses import asdict

    from oci.inference.vllm_server_pool import ManagedVLLMConfig

    config = plain_stage2_config_from_mapping(
        {
            "model": "primary",
            "vllm": {"gpus": ["cuda:0"], "extra_args": ["--max-model-len", "4096"]},
            "extraction_llm": {
                "model": "small",
                "endpoints": [{"endpoint": "http://e.test/v1", "max_concurrency": 2}],
            },
            "explicit_features": [_COMPLETE_EXPLICIT_FEATURE],
        },
        default_workers=2,
    )
    assert isinstance(config.vllm, ManagedVLLMConfig)
    # run_config.json stores the full dataclass, including runtime_* fields.
    snapshot = json.loads(json.dumps(asdict(config), default=str))
    snapshot.pop("runtime_response_format")
    assert plain_stage2_config_from_mapping(snapshot, default_workers=2) == config
    assert plain_stage2_config_from_mapping(config.public_dict(), default_workers=2) is not None
    root = Path(__file__).resolve().parents[1] / "example_configs"
    for path in sorted(root.glob("research_all_evidence*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))["stage2"]
        plain_stage2_config_from_mapping(
            {**raw, "endpoint": raw.get("endpoint") or "http://s.test/v1"}, default_workers=4
        )


def test_model_identity_resume_allows_endpoint_change_but_rejects_model_change(
    tmp_path: Path,
):
    completion = lambda _messages, _config: "{}"
    first = PlainHandoffStage2(
        config=PlainHandoffStage2Config(
            endpoint="http://old-endpoint.test/v1",
            model="same-model",
        ),
        clinical_question="Identify confounders.",
        completion=completion,
    )
    first._check_and_record_model_identity(tmp_path)

    moved = PlainHandoffStage2(
        config=PlainHandoffStage2Config(
            endpoint="http://new-endpoint.test/v1",
            model="same-model",
        ),
        clinical_question="Identify confounders.",
        completion=completion,
    )
    moved._check_and_record_model_identity(tmp_path)

    changed = PlainHandoffStage2(
        config=PlainHandoffStage2Config(
            endpoint="http://new-endpoint.test/v1",
            model="changed-model",
        ),
        clinical_question="Identify confounders.",
        completion=completion,
    )
    with pytest.raises(RuntimeError, match="actual running model identity changed"):
        changed._check_and_record_model_identity(tmp_path)


def test_stage2_extraction_forbids_multiple_patients_in_one_prompt(tmp_path: Path):
    definition = {
        "name": "performance_status",
        "value_type": "ordinal",
        "categories_or_unit": ["0", "1"],
    }
    with pytest.raises(ValueError, match="exactly one patient's record"):
        stage2_analysis._extraction_prompt(
            definitions=[definition],
            rows=[
                {"row_id": 0, "text": "ECOG 0."},
                {"row_id": 1, "text": "ECOG 1."},
            ],
        )

    prompt_row_ids = []

    def request_json(messages, validate, *, request_kind="interpretation"):
        assert request_kind == "extraction"
        body = prompt_inputs(messages)
        assert body["job"] == "extract_stage2_patient_variables"
        assert len(body["patients"]) == 1
        patient = body["patients"][0]
        prompt_row_ids.append(patient["row_id"])
        return validate(
            {
                "rows": [
                    {
                        "row_id": patient["row_id"],
                        "values": {
                            "performance_status": ("1" if "ECOG 1" in patient["text"] else "0")
                        },
                    }
                ]
            }
        )

    extracted = extract_rows(
        dataset=pd.DataFrame({"clinical_text": ["ECOG 0.", "ECOG 1.", "ECOG 0 again."]}),
        row_ids=[0, 1, 2],
        text_column="clinical_text",
        definitions=[definition],
        output_dir=tmp_path / "extraction",
        request_json=request_json,
        workers=3,
        max_prompt_chars=10_000,
    )

    assert sorted(prompt_row_ids) == [0, 1, 2]
    assert extracted["_oci_row_id"].tolist() == [0, 1, 2]


def test_stage2_serial_extraction_carries_state_across_lossless_token_chunks_and_resumes(
    tmp_path: Path,
):
    definitions = [
        {
            "name": f"feature_{index}",
            "description": "A clinical score.",
            "value_type": "continuous",
            "categories_or_unit": ["score"],
            "measurement_definition": "Extract the documented score.",
            "missing_value_rule": "Return null when undocumented.",
            "conflict_resolution": {
                "strategy": "latest",
                "positive_category": None,
            },
        }
        for index in range(3)
    ]
    note = ("a" * 450) + " EVIDENCE " + ("b" * 450)
    output = tmp_path / "serial"
    tokenizer = _CharacterChatTokenizer()
    bodies = []

    def request_json(messages, validate, *, request_kind="interpretation"):
        assert request_kind == "extraction"
        body = prompt_inputs(messages)
        assert body["job"] == "update_stage2_patient_variables_serially"
        bodies.append(body)
        patient = body["patient"]
        prior = patient["prior_extraction"]
        prior_state = patient["prior_feature_state"]
        values = {
            name: (
                int(name.removeprefix("feature_"))
                if "EVIDENCE" in patient["current_chunk"]
                else prior[name]
            )
            for name in prior
        }
        state = {
            name: (
                "documented in the EVIDENCE chunk"
                if "EVIDENCE" in patient["current_chunk"]
                else prior_state[name]
            )
            for name in prior
        }
        return validate(
            {
                "rows": [
                    {
                        "row_id": patient["row_id"],
                        "values": values,
                        "carry_forward_state": state,
                    }
                ]
            }
        )

    kwargs = {
        "dataset": pd.DataFrame({"clinical_text": [note]}),
        "row_ids": [0],
        "text_column": "clinical_text",
        "definitions": definitions,
        "output_dir": output,
        "workers": 1,
        "max_prompt_chars": 100_000,
        "feature_batch_size": 2,
        "tokenizer": tokenizer,
        "chunk_size_tokens": 200,
        "context_window_tokens": 6_000,
        "max_output_tokens": 500,
        "context_margin_tokens": 200,
    }
    frame = extract_rows(request_json=request_json, **kwargs)

    assert frame.loc[0, "feature_0"] == 0.0
    assert frame.loc[0, "feature_1"] == 1.0
    assert frame.loc[0, "feature_2"] == 2.0
    assert len(bodies) > 2
    assert all(
        body["patient"]["prior_extraction"][name] == int(name.removeprefix("feature_"))
        for body in bodies
        for name in body["patient"]["prior_extraction"]
        if body["patient"]["chunk"]["char_start"] > note.index("EVIDENCE") + len("EVIDENCE")
    )
    assert all(
        body["patient"]["prior_feature_state"][name]
        == "documented in the EVIDENCE chunk"
        for body in bodies
        for name in body["patient"]["prior_feature_state"]
        if body["patient"]["chunk"]["char_start"]
        > note.index("EVIDENCE") + len("EVIDENCE")
    )

    feature_dirs = sorted(
        (output / "batches" / "batch_00001" / "feature_batches").glob("batch_*")
    )
    assert len(feature_dirs) == 2
    for feature_dir in feature_dirs:
        inputs = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((feature_dir / "serial_chunks").glob("chunk_*/input.json"))
        ]
        assert "".join(value["chunk"]["text"] for value in inputs) == note
        assert all(value["chunk"]["source_tokens"] <= 200 for value in inputs)
        assert all(value["chunk"]["prompt_tokens"] <= 5_300 for value in inputs)
        manifest = json.loads(
            (feature_dir / "serial_extraction.json").read_text(encoding="utf-8")
        )
        assert manifest["lossless_source_coverage"] is True

    def unexpected_request(_messages, _validate, *, request_kind="interpretation"):
        raise AssertionError("completed serial chunks should be reused")

    parent_dir = output / "batches" / "batch_00001"
    (parent_dir / "complete.json").unlink()
    (parent_dir / "result.json").unlink()
    resumed = extract_rows(request_json=unexpected_request, **kwargs)
    pd.testing.assert_frame_equal(resumed, frame)


def test_extraction_request_exhaustion_aborts_without_scientific_checkpoint(
    tmp_path: Path,
):
    dataset = pd.DataFrame({"clinical_text": ["No usable response returned."]})
    definition = {
        "feature_id": "outer_001_feature_001",
        "name": "performance_status",
        "description": "Pretreatment performance status.",
        "value_type": "continuous",
        "categories_or_unit": ["ECOG score"],
        "measurement_definition": "Extract the documented ECOG score.",
        "missing_value_rule": "Return null when unavailable.",
    }
    output = tmp_path / "extraction"

    def exhausted_request(_messages, _validate, *, request_kind="interpretation"):
        assert request_kind == "extraction"
        raise stage2_analysis.Stage2RequestExhaustedError(
            "logical request deadline expired"
        )

    with pytest.raises(
        stage2_analysis.Stage2RequestExhaustedError,
        match="logical request deadline expired",
    ):
        extract_rows(
            dataset=dataset,
            row_ids=[0],
            text_column="clinical_text",
            definitions=[definition],
            output_dir=output,
            request_json=exhausted_request,
            workers=1,
            max_prompt_chars=10_000,
        )

    batch = output / "batches" / "batch_00001"
    assert not (batch / "extraction_failure.json").exists()
    assert not (batch / "result.json").exists()
    assert not (batch / "complete.json").exists()
    assert not (output / "complete.json").exists()


def test_discovery_keeps_each_card_separate_and_hides_provenance():
    packets = [
        {
            "packet_id": f"private-card-{index}",
            "architecture": "private-architecture",
            "observable_axes": ["outcome"],
            "content": {
                "score_summary": {"secret_statistic": 9.0},
                "representative_evidence": [{"text": text, "private_detail": "hidden"}],
            },
        }
        for index, text in enumerate(["Creatinine 1.2 mg/dL", "CT documents emphysema"])
    ]
    batches = stage2_workflow._partition_interpretation_packets(
        packets, architecture="private-architecture", max_prompt_chars=20_000,
    )
    assert batches == [[packets[0]], [packets[1]]]
    for batch in batches:
        messages = stage2_workflow._interpretation_prompt(
            architecture="private-architecture", packets=batch,
        )
        rendered = " ".join(message["content"] for message in messages)
        assert batch[0]["content"]["representative_evidence"][0]["text"] in rendered
        for forbidden in ("private-card", "private-architecture", "secret_statistic",
                          "private_detail", "supporting_items", "observable_axes"):
            assert forbidden not in rendered
    with pytest.raises(ValueError, match="exactly one evidence card"):
        stage2_workflow._interpretation_prompt(architecture="ignored", packets=packets)
    with pytest.raises(ValueError, match="cannot fit"):
        stage2_workflow._partition_interpretation_packets(
            packets, architecture="ignored", max_prompt_chars=100,
        )


def _fake_completion(calls):
    def complete(messages, _config):
        if messages[0]["content"].startswith("You identify clinical variables"):
            body = {"job": "infer_clinical_features_from_text_evidence"}
        else:
            body = prompt_inputs(messages)
        job = _prompt_job(body)
        calls.append(job)
        if job == "extract_stage2_patient_variables":
            rows = []
            for patient in body["patients"]:
                match = re.search(r"ECOG\s*([0-4])", patient["text"])
                values = {}
                for feature in body["features"]:
                    values[feature["name"]] = match.group(1) if match is not None else None
                rows.append({"row_id": patient["row_id"], "values": values})
            return json.dumps({"rows": rows})
        if job == "map_extracted_values_to_declared_category_ontology":
            return json.dumps(
                {
                    "corrections": [
                        {
                            "mapping_id": item["mapping_id"],
                            "value": f"ECOG {item['prior_extracted_value']}",
                        }
                        for item in body["items"]
                    ]
                }
            )
        if job == "review_stage2_small_model_extraction_ontology":
            return json.dumps(
                {
                    "action": "keep",
                    "reason": "Aggregate extraction values match the ontology.",
                }
            )
        if job == "infer_clinical_features_from_text_evidence":
            return json.dumps({"candidates": [{
                "name": "Performance status",
                "description": "Baseline functional performance status.",
                "basis": "The supplied text reports performance status.",
                "uncertainty": "",
            }]})
        if job == "consolidate_stage2_candidate_pool":
            assert "candidate_id" not in messages[1]["content"]
            return json.dumps({"merge_directives": []})
        assert job == "operationalize_stage2_candidate_group"
        assert "packet_id" not in messages[1]["content"]
        return json.dumps(
            {
                "description": "Baseline ECOG performance status.",
                "value_type": "ordinal",
                "categories_or_unit": ["ECOG 0", "ECOG 1", "ECOG 2", "ECOG 3", "ECOG 4"],
                "measurement_definition": "Extract the last pretreatment ECOG score.",
                "missing_value_rule": "Record undocumented separately from ECOG 0.",
                "stability_summary": "Supported in the supplied discovery contexts.",
                "caveats": "Resolve conflicting scores by date.",
            }
        )

    return complete


def test_stage2_pages_oversized_unicode_note_without_dropping_text(tmp_path: Path):
    # Oversize the source itself; Unicode serialization must not be what forces
    # an otherwise fitting note into pages.
    note = "start " + ("漢" * 8_000) + " pretreatment ECOG 2 end"
    dataset = pd.DataFrame({"clinical_text": [note]})
    definition = {
        "feature_id": "outer_001_feature_001",
        "name": "performance_status",
        "description": "Baseline ECOG performance status.",
        "value_type": "ordinal",
        "categories_or_unit": ["ECOG 0", "ECOG 1", "ECOG 2"],
        "roles": ["confounder"],
        "measurement_definition": "Extract the last pretreatment ECOG score.",
        "conflict_resolution": {"strategy": "mode"},
            "missing_value_rule": "Return null when undocumented.",
    }
    page_bodies = []
    prompt_sizes = []

    def request_json(messages, validate, *, request_kind="interpretation"):
        prompt_sizes.append(sum(len(message["content"]) for message in messages))
        body = prompt_inputs(messages)
        assert body["job"] == "extract_stage2_patient_variable_observations"
        patient = body["patient"]
        page_bodies.append(patient)
        observations = []
        if "ECOG 2" in patient["text"]:
            observations.append(
                _page_observation(
                    feature_name="performance_status",
                    value="ECOG 2",
                    text=patient["text"],
                    evidence="ECOG 2",
                )
            )
        response = {"rows": [{"row_id": patient["row_id"], "observations": observations}]}
        return validate(response)

    frame = extract_rows(
        dataset=dataset,
        row_ids=[0],
        text_column="clinical_text",
        definitions=[definition],
        output_dir=tmp_path / "extraction",
        request_json=request_json,
        workers=3,
        max_prompt_chars=5_000,
    )

    ordered_pages = sorted(page_bodies, key=lambda row: row["page"]["page_index"])
    assert "".join(row["text"] for row in ordered_pages) == note
    assert all(size <= 5_000 for size in prompt_sizes)
    assert frame.loc[0, "performance_status"] == "ECOG 2"
    decisions = json.loads(
        (
            tmp_path
            / "extraction"
            / "pages"
            / "row_00000000"
            / "reconciliation"
            / "decisions.json"
        ).read_text(encoding="utf-8")
    )
    decision = decisions["decisions"]["performance_status"]
    assert decision["resolution"] == "unanimous_value"
    assert decision["observations"][0]["evidence"] == "ECOG 2"
    assert note[
        decision["observations"][0]["source_start"] : decision["observations"][0][
            "source_end"
        ]
    ] == "ECOG 2"


def test_configured_feature_is_consolidated_with_discovery_and_keeps_supplied_ontology(
    tmp_path: Path,
):
    config = plain_stage2_config_from_mapping(
        {
            "endpoint": "http://stage2.test/v1",
            "model": "test-model",
            "explicit_features": [
                {
                    "name": "ecog_performance_status",
                    "description": "Pretreatment ECOG performance status.",
                    "value_type": "ordinal",
                    "categories_or_unit": ["0", "1", "2", "3", "4"],
                    "measurement_definition": (
                        "Extract the last explicitly documented pretreatment ECOG score."
                    ),
                    "missing_value_rule": "Return null when no score is documented.",
                    "roles": ["effect_modifier"],
                    "stability_summary": "Specified before Stage 2 discovery.",
                    "caveats": "Do not infer ECOG from symptoms.",
                }
            ],
        },
        default_workers=1,
    )
    assert config is not None
    jobs = []

    def completion(messages, _config):
        body = prompt_inputs(messages)
        job = _prompt_job(body)
        jobs.append(job)
        assert job == "consolidate_stage2_candidate_pool"
        assert body["configured_feature_names"] == ["ecog_performance_status"]
        return json.dumps(
            {
                "merge_directives": [
                    {
                        "inputs": [
                            "ecog_performance_status",
                            "performance_status",
                        ],
                        "output": "ecog_performance_status",
                    }
                ],
            }
        )

    runner = PlainHandoffStage2(
        config=config,
        clinical_question="Estimate treatment-effect heterogeneity.",
        completion=completion,
    )
    result = runner._consolidate_candidates(
        outer_fold=1,
        candidates=[
            {
                "candidate_id": "candidate_0001",
                "architecture": "architecture_alpha",
                "name": "performance_status",
                "description": "ECOG score found in Stage 1 evidence.",
                "value_type": "ambiguous",
                "supporting_packet_ids": ["packet_ecog"],
                "evidence_axes": ["treatment", "outcome"],
                "caveats": "",
            }
        ],
        evidence_packets=_original_evidence_packets(["packet_ecog"]),
        output_dir=tmp_path / "consolidation",
    )

    assert jobs == ["consolidate_stage2_candidate_pool"]
    assert len(result["features"]) == 1
    feature = result["features"][0]
    assert feature["name"] == "ecog_performance_status"
    assert feature["value_type"] == "ordinal"
    assert feature["categories_or_unit"] == ["0", "1", "2", "3", "4"]
    assert feature["measurement_definition"].startswith("Extract the last explicitly")
    assert feature["missing_value_rule"] == "Return null when no score is documented."
    # Configured roles and ontology remain authoritative even when the Stage 1
    # evidence would independently route the alias as a confounder.
    assert feature["roles"] == ["effect_modifier"]
    assert feature["supporting_packet_ids"] == ["packet_ecog"]
    assert feature["supporting_architectures"] == ["architecture_alpha"]
    assert feature["configured_explicit_feature"] is True
    assert (
        result["candidate_dispositions"]["configured_explicit_feature_0001"]["status"] == "retained"
    )
    assert result["candidate_dispositions"]["candidate_0001"]["status"] == "merged"
    provided = json.loads(
        next((tmp_path / "consolidation").rglob("provided_ontology.json")).read_text(
            encoding="utf-8"
        )
    )
    assert provided["status"] == "used_without_model_operationalization"

    exact_result = runner._consolidate_candidates(
        outer_fold=2,
        candidates=[
            {
                "candidate_id": "candidate_exact",
                "architecture": "architecture_beta",
                "name": "ecog_performance_status",
                "description": "The same normalized name returned by discovery.",
                "value_type": "ambiguous",
                "supporting_packet_ids": ["packet_exact"],
                "evidence_axes": ["outcome"],
                "caveats": "",
            }
        ],
        evidence_packets=_original_evidence_packets(["packet_exact"]),
    )
    assert jobs == ["consolidate_stage2_candidate_pool"]
    assert [feature["name"] for feature in exact_result["features"]] == ["ecog_performance_status"]


def test_global_consolidation_merges_aliases_and_retains_all_candidates_for_screening(
    tmp_path: Path,
):
    prompt_bodies = []
    evidence_packets = [
        {
            "packet_id": "packet_alpha",
            "content": {"representative_evidence": [{"text": "Blood glucose measured 105 mg/dL."}]},
        },
        {
            "packet_id": "packet_beta",
            "content": {
                "representative_evidence": [{"text": "Glycemia was documented numerically."}]
            },
        },
        {
            "packet_id": "packet_gamma",
            "content": {"representative_evidence": [{"text": "Resting pulse was 72 bpm."}]},
        },
        {
            "packet_id": "packet_delta",
            "content": {
                "representative_evidence": [
                    {"text": "James Lee had several unrelated chart findings."}
                ]
            },
        },
    ]

    def completion(messages, _config):
        rendered = messages[1]["content"]
        body = prompt_inputs(messages)
        prompt_bodies.append(body)
        job = _prompt_job(body)
        if job == "consolidate_stage2_candidate_pool":
            names = [feature["name"] for feature in body["features"]]
            if "glycemia" not in names:
                assert "james_lee_clinical_profile" in names
                return json.dumps({"merge_directives": []})
            assert names == [
                "blood_glucose_concentration",
                "glycemia",
                "heart_rate",
                "james_lee_clinical_profile",
            ]
            assert "clinical_question" not in body
            assert body["features"][-1]["descriptions"] == [
                "A named patient's multi-variable clinical profile."
            ]
            assert "candidate_id" not in rendered
            assert "group_id" not in rendered
            return json.dumps(
                {
                    "merge_directives": [
                        {
                            "inputs": ["blood_glucose_concentration", "glycemia"],
                            "output": "blood_glucose_concentration",
                        }
                    ],
                }
            )
        assert job == "operationalize_stage2_candidate_group"
        return json.dumps(
            {
                "description": body["candidate_feature_name"].replace("_", " "),
                "value_type": "continuous",
                "categories_or_unit": ["standard unit"],
                "measurement_definition": "Extract the last pretreatment scalar value.",
                "missing_value_rule": "Return null when undocumented.",
                "stability_summary": "Supported by candidate evidence.",
                "caveats": "",
            }
        )

    runner = PlainHandoffStage2(
        config=PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
        ),
        clinical_question="Estimate a treatment effect.",
        completion=completion,
    )
    candidates = [
        {
            "candidate_id": "candidate_0001",
            "architecture": "architecture_alpha",
            "name": "blood_glucose_concentration",
            "description": "Laboratory analyte A.",
            "value_type": "continuous",
            "supporting_packet_ids": ["packet_alpha"],
            "evidence_axes": ["treatment", "outcome"],
            "caveats": "",
        },
        {
            "candidate_id": "candidate_0002",
            "architecture": "architecture_beta",
            "name": "glycemia",
            "description": "Metabolic measurement B.",
            "value_type": "continuous",
            "supporting_packet_ids": ["packet_beta"],
            "evidence_axes": ["residual_effect"],
            "caveats": "",
        },
        {
            "candidate_id": "candidate_0003",
            "architecture": "architecture_gamma",
            "name": "heart_rate",
            "description": "Pretreatment heart rate.",
            "value_type": "continuous",
            "supporting_packet_ids": ["packet_gamma"],
            "evidence_axes": ["outcome"],
            "caveats": "",
        },
        {
            "candidate_id": "candidate_0004",
            "architecture": "architecture_delta",
            "name": "james_lee_clinical_profile",
            "description": "A named patient's multi-variable clinical profile.",
            "value_type": "ambiguous",
            "supporting_packet_ids": ["packet_delta"],
            "evidence_axes": ["unclear"],
            "caveats": "",
        },
    ]

    result = runner._consolidate_candidates(
        outer_fold=1,
        candidates=candidates,
        evidence_packets=evidence_packets,
        output_dir=tmp_path / "consolidation",
    )

    assert [feature["name"] for feature in result["features"]] == [
        "blood_glucose_concentration",
        "heart_rate",
        "james_lee_clinical_profile",
    ]
    assert result["features"][0]["supporting_packet_ids"] == [
        "packet_alpha",
        "packet_beta",
    ]
    assert [feature["roles"] for feature in result["features"]] == [[], [], []]
    assert result["candidate_dispositions"]["candidate_0001"]["status"] == "retained"
    assert result["candidate_dispositions"]["candidate_0002"]["status"] == "merged"
    assert result["candidate_dispositions"]["candidate_0003"]["status"] == "retained"
    assert result["candidate_dispositions"]["candidate_0004"]["status"] == "retained"
    assert [_prompt_job(body) for body in prompt_bodies].count(
        "consolidate_stage2_candidate_pool"
    ) == 2
    assert [_prompt_job(body) for body in prompt_bodies].count(
        "operationalize_stage2_candidate_group"
    ) == 3
    assert not (tmp_path / "consolidation" / "causal_role_filter.json").exists()
    ontology_bodies = {
        body["candidate_feature_name"]: body
        for body in prompt_bodies
        if _prompt_job(body) == "operationalize_stage2_candidate_group"
    }
    assert ontology_bodies["blood_glucose_concentration"]["supporting_evidence"] == [
        "Blood glucose measured 105 mg/dL.",
        "Glycemia was documented numerically.",
    ]
    assert ontology_bodies["heart_rate"]["supporting_evidence"] == ["Resting pulse was 72 bpm."]
    assert ontology_bodies["james_lee_clinical_profile"]["supporting_evidence"] == [
        "James Lee had several unrelated chart findings."
    ]


def test_llm_harmonizes_generic_mixed_values_and_applies_plan_to_heldout(
    tmp_path: Path,
):
    definition = {
        "feature_id": "feature_marker",
        "name": "tumor_marker_score",
        "description": "A pretreatment tumor marker score.",
        "value_type": "continuous",
        "categories_or_unit": ["points"],
        "modeling_strategy": "continuous_with_categorical_fallback",
        "roles": ["effect_modifier"],
        "measurement_definition": "Extract the documented pretreatment score.",
        "missing_value_rule": "Return null when undocumented.",
    }
    extracted = pd.DataFrame(
        {
            "_oci_row_id": [0, 1, 2, 3],
            "tumor_marker_score": [10.0, 60.0, "low", "high"],
        }
    )
    jobs = []

    def request_json(messages, validate, *, request_kind="interpretation"):
        assert request_kind == "interpretation"
        body = prompt_inputs(messages)
        jobs.append(body["job"])
        assert body["feature"]["name"] == "tumor_marker_score"
        return validate(
            {
                "target_representation": "categorical",
                "reason": "Text labels encode ranges, so common bins preserve meaning.",
                "canonical_categories": ["low", "high"],
                "categorical_value_map": [
                    {"raw_value": "low", "canonical_value": "low"},
                    {"raw_value": "high", "canonical_value": "high"},
                ],
                "numeric_bin_rules": [
                    {
                        "lower_bound": None,
                        "lower_inclusive": False,
                        "upper_bound": 50,
                        "upper_inclusive": False,
                        "canonical_value": "low",
                    },
                    {
                        "lower_bound": 50,
                        "lower_inclusive": True,
                        "upper_bound": None,
                        "upper_inclusive": False,
                        "canonical_value": "high",
                    },
                ],
            }
        )

    harmonized, definitions, report = stage2_analysis._harmonize_training_extraction(
        extracted=extracted,
        definitions=[definition],
        output_dir=tmp_path / "harmonization",
        request_json=request_json,
        max_prompt_chars=20_000,
    )

    assert jobs == ["harmonize_stage2_mixed_numeric_and_categorical_values"]
    assert definitions[0]["modeling_strategy"] == "categorical"
    assert definitions[0]["harmonization_plan"]["target_representation"] == ("categorical")
    assert harmonized["tumor_marker_score"].tolist() == ["low", "high", "low", "high"]
    assert report["plans_requested_from_llm"] == 1

    heldout = pd.DataFrame(
        {
            "_oci_row_id": [4, 5, 6],
            "tumor_marker_score": [75.0, "LOW", "unknown"],
        }
    )
    heldout_harmonized, audit = stage2_analysis._apply_harmonization_plans(
        heldout,
        definitions,
        scope="outer_heldout",
    )
    assert heldout_harmonized["tumor_marker_score"].tolist()[:2] == ["high", "low"]
    assert pd.isna(heldout_harmonized["tumor_marker_score"].iloc[2])
    assert audit["features"][0]["unmapped_nonmissing_rows"] == 1


def test_stage2_posthoc_oracle_ite_evaluation_uses_frozen_predictions(tmp_path: Path):
    prediction_path = tmp_path / "cross_fitted_predictions.csv"
    estimated = np.array([-0.20, -0.05, 0.10, 0.25, 0.40, 0.55])
    predictions = pd.DataFrame(
        {
            "_oci_row_id": list(range(6)),
            "outer_fold": [1, 1, 1, 2, 2, 2],
            "estimated_cate": estimated,
        }
    )
    predictions.to_csv(prediction_path, index=False)
    frozen_sha256 = stage2_workflow._file_sha256(prediction_path)
    dataset = pd.DataFrame({"true_ite_prob": 2.0 * estimated + 0.03})

    evaluation = stage2_workflow._evaluate_stage2_oracle_ite(
        prediction_path=prediction_path,
        dataset=dataset,
        output_dir=tmp_path,
    )

    assert evaluation["available"] is True
    assert evaluation["evaluation_is_post_hoc"] is True
    assert evaluation["all_outer_predictions_frozen_before_oracle_join"] is True
    assert evaluation["frozen_prediction_sha256"] == frozen_sha256
    assert evaluation["overall"]["pearson_correlation"] == pytest.approx(1.0)
    assert evaluation["overall"]["spearman_correlation"] == pytest.approx(1.0)
    assert len(evaluation["per_fold"]) == 2
    assert (tmp_path / "posthoc_oracle_ite_metrics.json").is_file()
    assert (tmp_path / "posthoc_predictions_with_oracle_ite.csv").is_file()
    assert "true_ite_prob" not in pd.read_csv(prediction_path).columns


def _tfidf_topic_payload(topic_id, terms):
    """Minimal real TF-IDF producer payload: one treatment-bank topic."""

    return {"discovery": {"topic_banks": {"treatment": {"topics": [{
        "topic_id": topic_id,
        "terms": [{"term": term, "loading": 0.9} for term in terms],
    }]}}}}


def _bow_outcome_payload(terms):
    """Minimal real text-model producer payload: one BoW outcome view."""

    return {"importance": {"views": [{
        "view_name": "bow",
        "outcome_positive": [{"feature": term, "coefficient": 0.5} for term in terms],
    }]}}


def test_plain_stage2_is_fold_scoped_and_resumable(tmp_path: Path):
    handoff = tmp_path / "handoff.jsonl"
    rows = [
        {
            "source": "tfidf",
            "outer_fold": 1,
            "inner_fold": None,
            "scope": "full_outer_train",
            "evidence": _tfidf_topic_payload(
                "treatment-ecog", ["ECOG", "poor performance status"]
            ),
        },
        {
            "source": "text_models",
            "outer_fold": 1,
            "inner_fold": 1,
            "scope": "candidate_consistency_inner_train",
            "evidence": _bow_outcome_payload(["ECOG 2 and unable to work"]),
        },
    ]
    handoff.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "stage2"
    calls = []
    config = PlainHandoffStage2Config(
        endpoint="http://stage2.test/v1",
        model="test-model",
        max_prompt_chars=8_000,
        workers=2,
        required_architectures=(),
    )

    first = run_plain_handoff_stage2(
        handoff_path=handoff,
        output_dir=output,
        clinical_question="Identify confounders.",
        config=config,
        completion=_fake_completion(calls),
    )

    assert first["outer_folds"] == 1
    assert first["features_by_fold"] == {"1": 1}
    definitions = json.loads(
        (output / "outer_001" / "feature_definitions.json").read_text(encoding="utf-8")
    )
    assert definitions["features"][0]["roles"] == []
    # Architecture-stratified compilation interprets the two producers
    # independently, then deterministically coalesces their exact shared name
    # before iterative semantic batches.
    assert calls.count("infer_clinical_features_from_text_evidence") == 2
    assert calls.count("consolidate_stage2_candidate_pool") == 1
    assert calls.count("operationalize_stage2_candidate_group") == 1

    second = run_plain_handoff_stage2(
        handoff_path=handoff,
        output_dir=output,
        clinical_question="Identify confounders.",
        config=config,
        completion=_fake_completion(calls),
    )

    assert second["features_by_fold"] == {"1": 1}
    assert len(calls) == 4

    rows[0]["evidence"] = _tfidf_topic_payload(
        "treatment-ecog", ["ECOG", "poor performance status", "newly compiled evidence"]
    )
    handoff.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(RuntimeError, match="different evidence plan"):
        run_plain_handoff_stage2(
            handoff_path=handoff,
            output_dir=output,
            clinical_question="Identify confounders.",
            config=config,
            completion=_fake_completion(calls),
        )


def test_plain_stage2_finishes_extraction_review_and_causal_estimation(
    tmp_path: Path,
    monkeypatch,
):
    handoff = tmp_path / "handoff.jsonl"
    evidence_rows = []
    for outer_fold in (1, 2):
        evidence_rows.append(
            {
                "source": "text_models",
                "outer_fold": outer_fold,
                "inner_fold": None,
                "scope": "full_outer_train",
                "evidence": {
                    **_bow_outcome_payload(["ECOG performance status"]),
                    # Text-model contexts record their fit/held-out sizes.
                    "metrics": {"handoff_fit_rows": 20, "handoff_heldout_rows": 20},
                },
            }
        )
        evidence_rows.append(
            {
                "source": "tfidf",
                "outer_fold": outer_fold,
                "inner_fold": None,
                "scope": "full_outer_train",
                "evidence": _tfidf_topic_payload("treatment-ecog", ["ECOG performance status"]),
            }
        )
    handoff.write_text(
        "".join(json.dumps(row) + "\n" for row in evidence_rows),
        encoding="utf-8",
    )
    dataset = pd.DataFrame(
        {
            "patient_id": [f"p{index:03d}" for index in range(40)],
            "clinical_text": [f"Pretreatment ECOG {index % 2}." for index in range(40)],
            "treatment_indicator": [int(index % 4 in {1, 3}) for index in range(40)],
            "outcome_indicator": [int(index % 5 in {0, 1}) for index in range(40)],
            "true_ite_prob": np.linspace(-0.2, 0.2, 40),
        }
    )
    split_path = tmp_path / "split_provenance.jsonl"
    split_rows = []
    for outer_fold, heldout in ((1, list(range(0, 20))), (2, list(range(20, 40)))):
        fit = sorted(set(range(40)) - set(heldout))
        split_rows.append(
            {
                "outer_fold": outer_fold,
                "fit_row_ids": fit,
                "heldout_row_ids": heldout,
                "inner_splits": [
                    {
                        "inner_fold": 1,
                        "fit_row_ids": fit[:10],
                        "heldout_row_ids": fit[10:],
                    },
                    {
                        "inner_fold": 2,
                        "fit_row_ids": fit[10:],
                        "heldout_row_ids": fit[:10],
                    },
                ],
            }
        )
    split_path.write_text(
        "".join(json.dumps(row) + "\n" for row in split_rows),
        encoding="utf-8",
    )
    from dataclasses import replace

    from oci.inference.stage2_multi_model_config import (
        ModifierCountConfig,
        ModifierSelectionConfig,
    )
    from tests.test_stage2_multi_model import small_policy

    # Exercise the real multi-model selector with the nested balanced_evidence
    # rule, a small numerical budget and a causal-forest final estimator.
    selection_policy = small_policy()
    selection_policy = replace(
        selection_policy,
        multi_model=replace(
            selection_policy.multi_model,
            modifier_selection=ModifierSelectionConfig(rule="balanced_evidence"),
            modifier_count=ModifierCountConfig(estimators=("causal_forest",)),
        ),
    )
    primary_calls = []
    extraction_calls = []
    output = tmp_path / "stage2"
    config = PlainHandoffStage2Config(
        endpoint="http://stage2.test/v1",
        model="test-model",
        max_prompt_chars=12_000,
        workers=4,
        statistical_selection=selection_policy,
        extraction_llm=stage2_workflow.Stage2ExtractionLLMConfig(
            endpoint="http://extract.test/v1",
            model="small-model",
            workers=4,
        ),
        max_review_rounds=1,
        estimation_trees=10,
        # Tiny synthetic cohorts: count-based informativeness rule with K=2.
        min_informative_patients=2,
        required_architectures=(),
        # The default super learner is exercised end to end by
        # tests/test_stage2_orthogonal_selection.py; this pipeline test keeps
        # the faster elastic-net nuisances.
        nuisance_learner="elastic_net",
    )

    first = run_plain_handoff_stage2(
        handoff_path=handoff,
        output_dir=output,
        clinical_question="Estimate the treatment effect.",
        config=config,
        completion=_fake_completion(primary_calls),
        extraction_completion=_fake_completion(extraction_calls),
        dataset=dataset,
        split_provenance_path=split_path,
        inner_folds=2,
        seed=7,
    )

    assert first["phase"] == "causal_estimation"
    assert first["causal_estimate"]["rows"] == 40
    assert first["causal_estimate"]["oracle_ite_evaluation"]["available"] is True
    assert "oracle_ite_pearson_correlation" in first["causal_estimate"]
    assert "oracle_ite_spearman_correlation" in first["causal_estimate"]
    assert (output / "cross_fitted_predictions.csv").is_file()
    assert (output / "causal_estimate.json").is_file()
    assert (output / "posthoc_oracle_ite_metrics.json").is_file()
    assert (output / "posthoc_predictions_with_oracle_ite.csv").is_file()
    # Oracle-free heterogeneity/overlap/balance diagnostics (P3-B).
    heterogeneity_path = output / "heterogeneity_diagnostics.json"
    assert str(heterogeneity_path) in first["artifacts"]
    heterogeneity_text = heterogeneity_path.read_text(encoding="utf-8")
    heterogeneity = json.loads(heterogeneity_text)
    assert heterogeneity["schema_version"] == (
        "stage2_heterogeneity_diagnostics_v5_unmeasured_confounding"
    )
    assert heterogeneity["oracle_columns_consumed"] is False
    assert "true_ite" not in heterogeneity_text
    assert heterogeneity["inputs"]["outer_folds"] == ["1", "2"]
    assert heterogeneity["inputs"]["rows"] == 40
    assert heterogeneity["inputs"]["recomputed_psi_max_abs_difference_from_aipw_score"] < 1e-9
    assert set(heterogeneity["pooling_rules"]) == {
        "blp", "gates", "clan", "rate", "overlap", "balance", "modifier_confirmation",
        "cate_variance_decomposition", "clip_sensitivity", "overlap_flags",
        "unmeasured_confounding_sensitivity",
    }
    # Reporting-only held-out confirmation also covers balanced_evidence selections.
    assert heterogeneity["modifier_confirmation"]["status"] == "ok"
    assert set(heterogeneity["modifier_confirmation"]["per_fold"]) == {"1", "2"}
    decomposition = heterogeneity["cate_variance_decomposition"]
    assert decomposition["status"] in {"ok", "unavailable"} and decomposition["interpretation"]
    assert set(decomposition["per_fold"]) == {"1", "2"}
    for name in (
        "blp", "gates", "clan", "rate", "overlap_weighted_ate", "ipw_weights", "trimming_curve"
    ):
        assert heterogeneity[name]["interpretation"]
        assert set(heterogeneity[name]["per_fold"]) == {"1", "2"}
        assert heterogeneity[name]["pooled"]["status"] in {"ok", "unavailable"}
    for name in ("overlap_weighted_ate", "ipw_weights", "trimming_curve"):
        assert heterogeneity[name]["pooled"]["status"] == "ok"
    assert [
        point["alpha"] for point in heterogeneity["trimming_curve"]["pooled"]["curve"]
    ] == [0.0, 0.01, 0.025, 0.05, 0.1, 0.15]
    assert heterogeneity["balance"]["interpretation"]
    assert heterogeneity["balance"]["training"]["status"] == "ok"
    assert heterogeneity["balance"]["training"]["candidate_scope"] in {
        "all_nuisance_candidates", "retained_features_only"
    }
    assert set(heterogeneity["balance"]["training"]["per_fold"]) == {"1", "2"}
    assert "pooled_summary" in heterogeneity["balance"]["training"]
    assert "pooled" in heterogeneity["balance"]["heldout"]
    assert "error" not in re.findall(r'"status": "([a-z_]+)"', heterogeneity_text)
    persisted_estimate = json.loads((output / "causal_estimate.json").read_text(encoding="utf-8"))
    heterogeneity_summary = persisted_estimate["heterogeneity_diagnostics"]
    assert heterogeneity_summary["path"] == str(heterogeneity_path)
    assert {
        "blp_pooled", "gates_pooled", "rate_pooled", "overlap_weighted_ate_pooled",
        "ipw_weights_pooled", "trimming_curve_pooled", "balance_training", "balance_heldout",
    } <= set(heterogeneity_summary)
    assert first["causal_estimate"]["heterogeneity_diagnostics"] == heterogeneity_summary
    # Existing documented keys are unchanged.
    assert {"ate", "standard_error", "mean_estimated_cate", "predictions_path"} <= set(
        persisted_estimate
    )
    # Co-primary estimands: the clipped AIPW ATE (lead by default) and the ATO.
    co_primary = persisted_estimate["co_primary_estimands"]
    assert co_primary["lead_estimand"] == "ate" and co_primary["order"] == ["ate", "ato"]
    assert co_primary["ate"]["estimate"] == pytest.approx(persisted_estimate["ate"])
    assert co_primary["ate"]["standard_error"] == pytest.approx(
        persisted_estimate["standard_error"])
    assert set(co_primary["ato"]["per_fold"]) == {"1", "2"}
    assert persisted_estimate["overlap_flags"]["bounds"] == [0.05, 0.95]
    cross_fitted = pd.read_csv(output / "cross_fitted_predictions.csv")
    assert cross_fitted["outside_overlap"].dtype == bool
    assert int(cross_fitted["outside_overlap"].sum()) == (
        persisted_estimate["overlap_flags"]["pooled"]["outside_overlap"])
    assert [p["clip"] for p in heterogeneity["clip_sensitivity"]["pooled"]["points"]] == [
        0.01, 0.02, 0.05]
    # Reporting-only unmeasured-confounding sensitivity: E-values, OVB bounds, benchmarks.
    sensitivity = heterogeneity["unmeasured_confounding_sensitivity"]
    assert sensitivity["status"] == "ok" and sensitivity["outcome_type"] == "binary"
    assert sensitivity["references"] and sensitivity["interpretation"]["evalue"]
    for estimand in ("ate", "ato"):
        assert set(sensitivity["evalues"][estimand]["per_fold"]) == {"1", "2"}
        assert set(sensitivity["omitted_variable_bias"][estimand]["per_fold"]) == {"1", "2"}
    ovb_ate = sensitivity["omitted_variable_bias"]["ate"]["pooled"]
    assert ovb_ate["status"] == "ok" and len(ovb_ate["contour"]) == 25
    assert ovb_ate["estimate"] == pytest.approx(persisted_estimate["ate"])
    assert sensitivity["benchmarks"]["status"] in {"ok", "unavailable"}
    for fold in ("1", "2"):
        benchmark_file = output / f"outer_00{fold}" / "estimation" / (
            "sensitivity_benchmarks") / "benchmarks.json"
        assert json.loads(benchmark_file.read_text(encoding="utf-8"))["status"] == "ok"
    compact = persisted_estimate["unmeasured_confounding_sensitivity"]
    assert compact == heterogeneity_summary["unmeasured_confounding_sensitivity"]
    assert compact["status"] == "ok" and {"ate", "ato", "benchmarks"} <= set(compact)
    selection_path = output / "outer_001" / "selection" / "elastic_net_selection.json"
    assert selection_path.is_file()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    assert selection["schema_version"] == (
        stage2_analysis.STAGE2_ROLE_SELECTION_SCHEMA_VERSION
    )
    assert selection["latent_construction"] == "disabled"
    assert selection["pairwise_association_screen"] == "disabled"
    assert selection["cross_fitted_nuisance_models"][
        "predictions_are_inner_fold_out_of_fold"
    ] is True
    assert selection["selection_authority"] == "multi_model"
    assert selection["final_role_assignment"] == (
        "predictive_nuisances_and_nested_r_loss_modifiers"
    )
    assert selection["role_adjudication"]["llm_requests"] == 0
    assert selection["role_adjudication"]["llm_roles_are_binding"] is False
    statistical_evidence_path = (
        output
        / "outer_001"
        / "selection"
        / "statistical_evidence.json"
    )
    assert statistical_evidence_path.is_file()
    statistical_evidence = statistical_evidence_path.read_text(encoding="utf-8").casefold()
    assert "true_ite" not in statistical_evidence
    assert "generation_config" not in statistical_evidence
    assert not (output / "outer_001" / "selection" / "role_adjudication").exists()
    assert not (output / "outer_001" / "selection" / "stage2_evidence").exists()
    assert (
        output
        / "outer_001"
        / "ontology_supervision"
        / "round_001"
        / "supervisor"
        / "complete.json"
    ).is_file()
    assert (output / "outer_001" / "extraction" / "extracted_features.csv").is_file()
    assert (output / "outer_001" / "estimation" / "predictions.csv").is_file()
    diagnostics = json.loads(
        (output / "outer_001" / "estimation" / "diagnostics.json").read_text(
            encoding="utf-8"
        )
    )
    assert diagnostics["model_family"] == "causal_forest_dml"
    assert diagnostics["causal_forest_fit_audit"]["outcome_model_contract"] == {
        "outcome_type": "binary",
        "discrete_outcome": True,
        "model_class": (
            "oci.models.elastic_net_nuisance.ElasticNetLogisticClassifier"
        ),
        "prediction_interface": "predict_proba",
        "criterion": "log_loss",
        "penalty": "elastic_net",
    }
    assert diagnostics["nuisance_model_family"] == "elastic_net"
    assert "extract_stage2_patient_variables" not in primary_calls
    assert extraction_calls
    assert set(extraction_calls) == {"extract_stage2_patient_variables"}
    calls_after_first = (len(primary_calls), len(extraction_calls))
    extraction_path = output / "outer_001" / "extraction" / "extracted_features.csv"
    selection_bytes = selection_path.read_bytes()
    extraction_bytes = extraction_path.read_bytes()
    estimation_complete_path = output / "outer_001" / "estimation" / "complete.json"
    old_estimation_completion = json.loads(
        estimation_complete_path.read_text(encoding="utf-8")
    )
    old_estimation_completion["schema_version"] = (
        "stage2_outer_estimation_v4_causal_forest"
    )
    estimation_complete_path.write_text(
        json.dumps(old_estimation_completion),
        encoding="utf-8",
    )
    forest_fits = []
    original_forest_fit = stage2_analysis.CausalForestHead.fit

    def tracked_forest_fit(self, *args, **kwargs):
        forest_fits.append(self.outcome_type)
        return original_forest_fit(self, *args, **kwargs)

    monkeypatch.setattr(stage2_analysis.CausalForestHead, "fit", tracked_forest_fit)

    second = run_plain_handoff_stage2(
        handoff_path=handoff,
        output_dir=output,
        clinical_question="Estimate the treatment effect.",
        config=config,
        completion=_fake_completion(primary_calls),
        extraction_completion=_fake_completion(extraction_calls),
        dataset=dataset,
        split_provenance_path=split_path,
        inner_folds=2,
        seed=7,
    )

    assert second["causal_estimate"]["rows"] == 40
    assert (len(primary_calls), len(extraction_calls)) == calls_after_first
    assert forest_fits == ["binary"]
    assert selection_path.read_bytes() == selection_bytes
    assert extraction_path.read_bytes() == extraction_bytes
    resumed_estimation_completion = json.loads(
        estimation_complete_path.read_text(encoding="utf-8")
    )
    assert resumed_estimation_completion["schema_version"] == (
        stage2_analysis.ESTIMATION_CHECKPOINT_SCHEMA_VERSION
    )


def test_repeated_training_extraction_failures_refine_ontology_and_reextract(
    tmp_path: Path,
    monkeypatch,
):
    from oci.inference import stage2_modifier_count

    # Six training patients cannot support numerical selection; this test is
    # about extraction-failure ontology refinement, so retain the candidate.
    monkeypatch.setattr(
        stage2_analysis,
        "select_stage2_features_multi_model",
        lambda **kwargs: (
            [{**kwargs["definitions"][0], "nuisance_model_roles": ["treatment", "outcome"]}],
            {"schema_version": stage2_analysis.STAGE2_ROLE_SELECTION_SCHEMA_VERSION},
            [],
            [],
        ),
    )
    monkeypatch.setattr(
        stage2_modifier_count,
        "select_modifier_count",
        lambda **kw: (kw["selected"], kw["role_report"], {"status": "complete"}),
    )
    dataset = pd.DataFrame(
        {
            "patient_id": [f"p{index:02d}" for index in range(12)],
            "clinical_text": [
                f"Pretreatment overall stage {'III' if index % 2 == 0 else 'IV'}."
                for index in range(12)
            ],
            "treatment_indicator": [index % 2 for index in range(12)],
            "outcome_indicator": [(index // 2) % 2 for index in range(12)],
        }
    )
    definition = {
        "feature_id": "outer_001_feature_001",
        "name": "disease_stage",
        "description": "Pretreatment overall disease stage.",
        "value_type": "ordinal",
        "categories_or_unit": ["stage_i", "stage_ii"],
        "roles": ["confounder"],
        "measurement_definition": "Extract the documented overall stage.",
        "missing_value_rule": "Use null when no overall stage is documented.",
    }
    fit_ids = list(range(6))
    split = {
        "outer_fold": 1,
        "fit_row_ids": fit_ids,
        "heldout_row_ids": list(range(6, 12)),
        "inner_splits": [
            {"inner_fold": 1, "fit_row_ids": fit_ids[:3], "heldout_row_ids": fit_ids[3:]},
            {"inner_fold": 2, "fit_row_ids": fit_ids[3:], "heldout_row_ids": fit_ids[:3]},
        ],
    }
    jobs = []

    def request_json(messages, validate, *, request_kind="interpretation", initial_reasoning_effort=None):
        body = prompt_inputs(messages)
        jobs.append(body["job"])
        if body["job"] == "extract_stage2_patient_variables":
            categories = body["features"][0]["categories_or_unit"]
            rows = []
            for patient in body["patients"]:
                raw_stage = "stage_iii" if "stage III" in patient["text"] else "stage_iv"
                rows.append(
                    {
                        "row_id": patient["row_id"],
                        "values": {
                            "disease_stage": (
                                raw_stage
                                if raw_stage in categories
                                else raw_stage.replace("stage_", "TNM stage ").upper()
                            )
                        },
                    }
                )
            return validate({"rows": rows})
        if body["job"] == "map_extracted_values_to_declared_category_ontology":
            return validate(
                {
                    "corrections": [
                        {"mapping_id": item["mapping_id"], "value": None} for item in body["items"]
                    ]
                }
            )
        if body["job"] == "refine_stage2_feature_ontology_from_repeated_extraction_failures":
            assert body["feature"]["name"] == "disease_stage"
            assert body["repeated_failure_patterns"][0]["patient_count"] == 6
            assert "patients" not in body
            return validate(
                {
                    "action": "revise",
                    "reason": "The initial closed ontology omitted supported stage groups.",
                    "description": "Pretreatment overall disease stage group.",
                    "value_type": "ordinal",
                    "categories_or_unit": ["stage_iii", "stage_iv"],
                    "measurement_definition": (
                        "Extract the explicitly documented pretreatment overall stage group."
                    ),
                    "missing_value_rule": (
                        "Use null when no pretreatment overall stage group is documented."
                    ),
                }
            )
        return validate(
            {
                "action": "keep",
                "reason": "The refined ontology extracted consistently.",
            }
        )

    output = tmp_path / "outer_001"
    result = run_fold_analysis(
        dataset=dataset,
        definitions=[definition],
        split=split,
        clinical_question="Estimate treatment effect.",
        unit_id_column="patient_id",
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        outcome_type="binary",
        inner_folds=2,
        seed=23,
        output_dir=output,
        request_json=request_json,
        config=PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
            workers=1,
            max_review_rounds=1,
            ontology_refinement_min_failure_patients=3,
            max_ontology_refinement_rounds=2,
            estimation_trees=10,
            # Tiny synthetic cohorts: count-based informativeness rule with K=2.
            min_informative_patients=2,
        ),
    )

    assert result["ontology_refinement_rounds"] == 1
    assert jobs.count("refine_stage2_feature_ontology_from_repeated_extraction_failures") == 1
    initial_summary = json.loads(
        (
            output
            / "ontology_supervision"
            / "round_001"
            / "extraction"
            / "failure_summary.json"
        ).read_text(encoding="utf-8")
    )
    assert initial_summary["feature_failure_patterns"][0]["patient_count"] == 6
    feedback = json.loads(
        (
            output
            / "ontology_supervision"
            / "round_001"
            / "failure_ontology_refinement"
            / "complete.json"
        ).read_text(encoding="utf-8")
    )
    assert feedback["stopped_reason"] == "no_repeated_feature_failures"
    refined = pd.read_csv(
        output
        / "ontology_supervision"
        / "round_001"
        / "failure_ontology_refinement"
        / "round_001"
        / "extraction"
        / "extracted.csv"
    )
    assert refined["disease_stage"].notna().all()
    supervisor_input = json.loads(
        (
            output
            / "ontology_supervision"
            / "round_001"
            / "supervisor"
            / "feature_0001"
            / "input.json"
        ).read_text(encoding="utf-8")
    )
    assert supervisor_input["feature"]["categories_or_unit"] == ["stage_iii", "stage_iv"]


def test_failure_driven_ontology_refinement_never_rewrites_explicit_feature(
    tmp_path: Path,
):
    feature = {
        "feature_id": "outer_001_feature_001",
        "name": "investigator_marker",
        "description": "Investigator-specified marker.",
        "value_type": "binary",
        "categories_or_unit": ["absent", "present"],
        "measurement_definition": "Extract the documented marker status.",
        "missing_value_rule": "Use null when undocumented.",
        "roles": ["effect_modifier"],
        "configured_explicit_feature": True,
    }

    def unexpected_request(_messages, _validate, *, request_kind="interpretation"):
        raise AssertionError("explicit ontology must not be sent for refinement")

    updated, changed, report = stage2_analysis._request_ontology_refinements(
        definitions=[feature],
        repeated_patterns={
            "investigator_marker": [
                {
                    "feature_name": "investigator_marker",
                    "failure_kind": "out_of_ontology_category",
                    "reason": "outside ontology",
                    "patient_count": 5,
                    "patient_row_ids": [1, 2, 3, 4, 5],
                    "example_values": ["equivocal"],
                    "allowed_categories": ["absent", "present"],
                }
            ]
        },
        output_dir=tmp_path / "ontology_refinement",
        request_json=unexpected_request,
        workers=4,
    )

    assert changed is False
    assert updated == [feature]
    assert report["model_requested_features"] == 0
    assert report["immutable_explicit_features"] == 1
