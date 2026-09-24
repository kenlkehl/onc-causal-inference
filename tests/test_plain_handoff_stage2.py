from __future__ import annotations

from tests.stage2_prompt_spy import prompt_inputs

import json
import re
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import oci.inference.plain_handoff_stage2 as stage2_workflow
import oci.inference.plain_handoff_stage2_analysis as stage2_analysis

from oci.inference.plain_handoff_stage2 import (
    PlainHandoffStage2,
    PlainHandoffStage2Config,
    Stage2ExtractionLLMConfig,
    packetize_handoff,
    plain_stage2_config_from_mapping,
    run_plain_handoff_stage2,
)
from oci.inference.plain_handoff_stage2_analysis import extract_rows, run_fold_analysis
from oci.inference.stage2_role_adjudication import Stage2RoleAdjudicationConfig


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
    if body.get("task") == "analyze_cluster":
        return "analyze_stage2_variable_cluster"
    if body.get("task") == "outer_fold_role_adjudication":
        return "adjudicate_stage2_variable_role"
    if body.get("task") == "adjudicate_stage2_roles_from_all_evidence":
        return "adjudicate_stage2_roles_from_all_evidence"
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


def test_stage2_config_allows_endpoint_without_model():
    config = plain_stage2_config_from_mapping(
        {"endpoint": "http://stage2.test/v1"},
        default_workers=8,
    )

    assert config is not None
    assert config.endpoint == "http://stage2.test/v1"
    assert config.model == ""
    assert config.request_timeout == 7200.0
    assert config.request_attempt_timeout == 900.0
    assert config.transport_max_attempts == 6
    assert config.max_response_repairs == 15
    assert config.thinking_after_response_repairs == 5
    assert config.max_tokens == 100_000
    assert config.extraction_max_tokens == 75_000
    assert config.repetition_penalty is None
    assert config.interpretation_reasoning_effort == "auto"
    assert config.extraction_reasoning_effort == "auto"
    assert config.max_prompt_chars == 100_000
    assert config.consolidation_max_prompt_chars == 640_000
    assert config.operationalization_max_prompt_chars == 640_000
    assert config.consolidation_batch_size == 20
    assert config.consolidation_alphabetical_rounds == 5
    assert config.consolidation_max_rounds == 55
    assert (
        config.consolidation_max_rounds - config.consolidation_alphabetical_rounds
        == stage2_workflow.DEFAULT_CONSOLIDATION_SHUFFLE_ROUNDS
        == 50
    )
    assert config.extraction_max_prompt_chars == 640_000
    assert config.extraction_feature_batch_size == 10
    assert config.extraction_chunk_size_tokens == 50_000
    assert config.extraction_context_window_tokens == 131_072
    assert config.extraction_context_margin_tokens == 1_024
    assert config.vllm_rapid_switch_seconds == 900.0
    assert config.ontology_refinement_min_failure_patients == 3
    assert config.max_ontology_refinement_rounds == 2
    assert config.evidence_compiler == "semantic_cluster_cards_v2"
    assert config.evidence_max_cards_per_fold == 400
    assert config.extraction_llm is None
    assert config.input_temporal_scope == "pre_index_treatment"
    assert config.agentic_selection.cluster_similarity_threshold == 0.60
    assert config.agentic_selection.cluster_consensus_fraction == 0.60
    assert config.agentic_selection.max_latents_per_cluster == 2
    assert config.selection_consolidation.enabled is False
    assert config.selection_consolidation.neighbor_count == 10
    assert config.selection_consolidation.minimum_pairwise_association == 0.85
    assert config.workers == 4


def test_stage2_config_parses_agentic_evidence_pair_chunk_size():
    config = plain_stage2_config_from_mapping(
        {
            "endpoint": "http://stage2.test/v1",
            "agentic_evidence_pair_chunk_size": 37,
        },
        default_workers=8,
    )

    assert config is not None
    assert config.agentic_evidence_pair_chunk_size == 37
    with pytest.raises(ValueError, match="pair_chunk_size"):
        PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            agentic_evidence_pair_chunk_size=0,
        ).validate(require_model=False)


def test_extraction_token_cap_does_not_invalidate_completed_feature_definitions():
    prior_config = PlainHandoffStage2Config(
        endpoint="http://stage2.test/v1",
        model="primary-model",
        extraction_max_tokens=100_000,
    )
    resumed_config = PlainHandoffStage2Config(
        endpoint="http://stage2.test/v1",
        model="primary-model",
        extraction_max_tokens=60_000,
    )

    def definition_inputs(config):
        return stage2_workflow._feature_definition_input_value(
            config=config,
            clinical_question="Identify confounders.",
            outer_fold=1,
            discovery_packets=[],
            seed=42,
        )

    assert definition_inputs(resumed_config) == definition_inputs(prior_config)
    assert stage2_workflow._stage2_request_policy(
        prior_config,
        "extraction",
    )["max_tokens"] == 100_000
    assert stage2_workflow._stage2_request_policy(
        resumed_config,
        "extraction",
    )["max_tokens"] == 60_000


def test_agentic_selection_policy_does_not_invalidate_feature_definitions():
    prior_config = PlainHandoffStage2Config(
        endpoint="http://stage2.test/v1",
        model="primary-model",
        agentic_selection=stage2_workflow.Stage2AgenticSelectionConfig(
            cluster_similarity_threshold=0.60,
        ),
    )
    resumed_config = PlainHandoffStage2Config(
        endpoint="http://stage2.test/v1",
        model="primary-model",
        agentic_selection=stage2_workflow.Stage2AgenticSelectionConfig(
            cluster_similarity_threshold=0.75,
        ),
    )

    def definition_inputs(config):
        return stage2_workflow._feature_definition_input_value(
            config=config,
            clinical_question="Identify confounders.",
            outer_fold=1,
            discovery_packets=[],
            seed=42,
        )

    assert definition_inputs(resumed_config) == definition_inputs(prior_config)


def test_stage2_analysis_defaults_pre_refinement_config_fields(caplog):
    class PreRefinementConfig:
        pass

    limits = stage2_analysis._ontology_refinement_limits(PreRefinementConfig())

    assert limits == (3, 2)
    assert "pre-ontology-refinement config" in caplog.text


def test_large_statistical_selector_runs_in_loky_process(monkeypatch):
    expected = ([{"feature_id": "selected"}], {"status": "complete"}, [], [])
    calls = []

    def fake_selector(**arguments):
        calls.append(arguments)
        return expected

    class FakeFuture:
        def result(self):
            return expected

    class FakeExecutor:
        def __init__(self, *, max_workers):
            assert max_workers == 1

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def submit(self, function, **arguments):
            assert function is fake_selector
            calls.append(arguments)
            return FakeFuture()

    monkeypatch.setattr(
        stage2_analysis,
        "STATISTICAL_SELECTION_PROCESS_ISOLATION_MIN_CANDIDATES",
        2,
    )
    monkeypatch.setattr(
        stage2_analysis,
        "select_stage2_features_elastic_net",
        fake_selector,
    )
    monkeypatch.setattr(
        stage2_analysis,
        "LokyProcessPoolExecutor",
        FakeExecutor,
    )

    result = stage2_analysis._run_stage2_statistical_selection(
        {"definitions": [{"feature_id": "one"}, {"feature_id": "two"}]}
    )

    assert result == expected
    assert len(calls) == 1


def test_stage2_config_parses_independent_large_context_prompt_budgets():
    config = plain_stage2_config_from_mapping(
        {
            "endpoint": "http://stage2.test/v1",
            "request_timeout": 1_200,
            "request_attempt_timeout": 120,
            "max_tokens": 148_000,
            "extraction_max_tokens": 72_000,
            "max_response_repairs": 8,
            "thinking_after_response_repairs": 3,
            "repetition_penalty": 1.15,
            "interpretation_reasoning_effort": "medium",
            "extraction_reasoning_effort": "none",
            "max_prompt_chars": 90_000,
            "consolidation_max_prompt_chars": 450_000,
            "operationalization_max_prompt_chars": 475_000,
            "consolidation_batch_size": 18,
            "consolidation_alphabetical_rounds": 4,
            "consolidation_max_rounds": 7,
            "extraction_max_prompt_chars": 500_000,
            "extraction_feature_batch_size": 7,
            "extraction_chunk_size_tokens": 41_000,
            "extraction_context_window_tokens": 160_000,
            "extraction_context_margin_tokens": 2_000,
            "vllm_rapid_switch_seconds": 1_200,
            "extraction_llm": {
                "endpoint": "http://extract.test/v1",
                "model": "small-model",
                "api_key": "secret-small-key",
                "workers": 3,
            },
            "ontology_refinement_min_failure_patients": 4,
            "max_ontology_refinement_rounds": 3,
            "input_temporal_scope": "pre_index_treatment",
            "agentic_selection": {
                "cluster_similarity_threshold": 0.7,
                "cluster_consensus_fraction": 0.8,
                "cluster_max_size": 10,
                "max_latents_per_cluster": 2,
                "cluster_tool_call_limit": 6,
                "adjudicator_tool_call_limit": 7,
            },
            "selection_consolidation": {
                "enabled": True,
                "neighbor_count": 8,
                "embedding_model": "test-embedding-model",
                "embedding_device": "cpu",
                "max_latents_per_cluster": 3,
                "minimum_pairwise_association": 0.9,
                "latent_min_coverage": 0.1,
            },
            "role_adjudication": {
                "enabled": True,
                "max_description_chars": 900,
                "max_measurement_definition_chars": 1_100,
                "max_categories_per_feature": 25,
                "max_candidates_per_request": 17,
            },
        },
        default_workers=1,
    )

    assert config is not None
    assert config.request_timeout == 1_200.0
    assert config.request_attempt_timeout == 120.0
    assert config.max_tokens == 148_000
    assert config.extraction_max_tokens == 72_000
    assert config.max_response_repairs == 8
    assert config.thinking_after_response_repairs == 3
    assert config.repetition_penalty == 1.15
    assert config.interpretation_reasoning_effort == "medium"
    assert config.extraction_reasoning_effort == "none"
    assert config.max_prompt_chars == 90_000
    assert config.consolidation_max_prompt_chars == 450_000
    assert config.operationalization_max_prompt_chars == 475_000
    assert config.consolidation_batch_size == 18
    assert config.consolidation_alphabetical_rounds == 4
    assert config.consolidation_max_rounds == 7
    assert config.extraction_max_prompt_chars == 500_000
    assert config.extraction_feature_batch_size == 7
    assert config.extraction_chunk_size_tokens == 41_000
    assert config.extraction_context_window_tokens == 160_000
    assert config.extraction_context_margin_tokens == 2_000
    assert config.vllm_rapid_switch_seconds == 1_200.0
    assert config.ontology_refinement_min_failure_patients == 4
    assert config.max_ontology_refinement_rounds == 3
    assert config.extraction_llm.endpoint == "http://extract.test/v1"
    assert config.extraction_llm.model == "small-model"
    assert config.extraction_llm.workers == 3
    assert config.input_temporal_scope == "pre_index_treatment"
    assert config.agentic_selection.cluster_similarity_threshold == 0.7
    assert config.agentic_selection.cluster_consensus_fraction == 0.8
    assert config.agentic_selection.cluster_max_size == 10
    assert config.agentic_selection.cluster_tool_call_limit == 6
    assert config.selection_consolidation.enabled is True
    assert config.selection_consolidation.neighbor_count == 8
    assert config.selection_consolidation.embedding_model == "test-embedding-model"
    assert config.selection_consolidation.max_latents_per_cluster == 3
    assert config.selection_consolidation.minimum_pairwise_association == 0.9
    assert config.selection_consolidation.latent_min_coverage == 0.1
    assert config.role_adjudication.enabled is True
    assert config.role_adjudication.max_description_chars == 900
    assert config.role_adjudication.max_measurement_definition_chars == 1_100
    assert config.role_adjudication.max_categories_per_feature == 25
    assert config.role_adjudication.max_candidates_per_request == 17
    assert config.public_dict()["max_tokens"] == 148_000
    assert config.public_dict()["request_timeout"] == 1_200.0
    assert config.public_dict()["request_attempt_timeout"] == 120.0
    assert config.public_dict()["extraction_max_tokens"] == 72_000
    assert config.public_dict()["max_response_repairs"] == 8
    assert config.public_dict()["thinking_after_response_repairs"] == 3
    assert config.public_dict()["repetition_penalty"] == 1.15
    assert config.public_dict()["interpretation_reasoning_effort"] == "medium"
    assert config.public_dict()["extraction_reasoning_effort"] == "none"
    assert config.public_dict()["consolidation_max_prompt_chars"] == 450_000
    assert config.public_dict()["operationalization_max_prompt_chars"] == 475_000
    assert config.public_dict()["consolidation_batch_size"] == 18
    assert config.public_dict()["consolidation_alphabetical_rounds"] == 4
    assert config.public_dict()["consolidation_max_rounds"] == 7
    assert config.public_dict()["extraction_max_prompt_chars"] == 500_000
    assert config.public_dict()["extraction_feature_batch_size"] == 7
    assert config.public_dict()["extraction_chunk_size_tokens"] == 41_000
    assert config.public_dict()["extraction_context_window_tokens"] == 160_000
    assert config.public_dict()["extraction_context_margin_tokens"] == 2_000
    assert config.public_dict()["vllm_rapid_switch_seconds"] == 1_200.0
    assert config.public_dict()["ontology_refinement_min_failure_patients"] == 4
    assert config.public_dict()["max_ontology_refinement_rounds"] == 3
    assert config.public_dict()["role_adjudication"] == {
        "enabled": True,
        "max_description_chars": 900,
        "max_measurement_definition_chars": 1_100,
        "max_categories_per_feature": 25,
        "max_candidates_per_request": 17,
    }
    assert config.public_dict()["extraction_llm"]["api_key"] == "<redacted>"
    assert config.public_dict()["api_key"] == "<redacted>"


def test_stage2_allows_primary_and_extraction_models_on_the_same_endpoint():
    config = plain_stage2_config_from_mapping(
        {
            "endpoint": "http://stage2.test/v1",
            "model": "large-model",
            "extraction_llm": {
                "endpoint": "http://stage2.test/v1/",
                "model": "small-model",
            },
        },
        default_workers=1,
    )

    assert config is not None
    assert config.endpoint == config.extraction_llm.endpoint


def test_stage2_config_warns_and_ignores_retired_colbert_settings(caplog):
    config = plain_stage2_config_from_mapping(
        {
            "endpoint": "http://stage2.test/v1",
            "evidence_community_enabled": True,
            "candidate_selection_method": "late_interaction",
            "max_evaluation_rounds": 10,
            "screening_trees": 200,
        },
        default_workers=1,
    )

    assert config is not None
    public = config.public_dict()
    assert "evidence_community_enabled" not in public
    assert "candidate_selection_method" not in public
    assert "max_evaluation_rounds" not in public
    assert "ignoring retired Stage 2" in caplog.text


def test_extraction_runtime_continuation_route_is_parsed_and_audited_separately():
    config = plain_stage2_config_from_mapping(
        {
            "endpoint": "http://primary.test/v1",
            "model": "primary-model",
            "extraction_llm": {
                "endpoint": "http://original-extractor.test/v1",
                "model": "original-extractor",
                "workers": 3,
                "runtime_endpoint": "http://replacement-extractor.test/v1/",
                "runtime_model": "replacement-extractor",
                "runtime_api_key": "secret",
            },
        },
        default_workers=1,
    )

    assert config is not None
    assert config.extraction_llm is not None
    assert config.extraction_llm.model == "original-extractor"
    assert config.extraction_llm.runtime_endpoint == (
        "http://replacement-extractor.test/v1"
    )
    assert config.extraction_llm.runtime_model == "replacement-extractor"
    public = config.extraction_llm.public_dict()
    assert public["model"] == "original-extractor"
    assert public["runtime_model"] == "replacement-extractor"
    assert public["runtime_api_key"] == "<redacted>"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"runtime_endpoint": "http://replacement.test/v1"},
        {"runtime_model": "replacement-model"},
        {"runtime_endpoint": "not-a-url", "runtime_model": "replacement-model"},
    ],
)
def test_extraction_runtime_continuation_route_requires_endpoint_model_pair(kwargs):
    with pytest.raises(ValueError, match="runtime_"):
        Stage2ExtractionLLMConfig(
            endpoint="http://original.test/v1",
            model="original-model",
            **kwargs,
        ).validate()


@pytest.mark.parametrize(
    "field_name,value",
    [
        ("cluster_similarity_threshold", 0.0),
        ("cluster_consensus_fraction", 1.1),
        ("missingness_weight", float("nan")),
        ("cluster_max_size", 1),
        ("cluster_tool_call_limit", 0),
    ],
)
def test_stage2_config_rejects_invalid_agentic_selection_policy(field_name, value):
    policy = stage2_workflow.Stage2AgenticSelectionConfig(**{field_name: value})
    with pytest.raises(ValueError, match=field_name):
        PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
            agentic_selection=policy,
        ).validate()


@pytest.mark.parametrize(
    "removed_key",
    [
        "selection_workers",
        "confounder_p_value_threshold",
        "confounder_min_inner_fold_fraction",
        "effect_modifier_p_value_threshold",
        "effect_modifier_min_inner_fold_fraction",
    ],
)
def test_stage2_config_migrates_retired_p_value_screen_settings(removed_key, caplog):
    config = plain_stage2_config_from_mapping(
        {"endpoint": "http://stage2.test/v1", removed_key: 1},
        default_workers=1,
    )

    assert config is not None
    assert "migrating legacy Stage 2 screen settings" in caplog.text
    assert (
        config.statistical_selection.nuisance_selection_rule
        == "any_inner_fold_union"
    )
    assert (
        config.statistical_selection.modifier_selection_rule
        == "any_inner_fold_union"
    )


@pytest.mark.parametrize(
    "rapid_switch_seconds",
    [-1, True, float("inf"), float("nan"), "900"],
)
def test_stage2_config_rejects_invalid_vllm_rapid_switch_seconds(
    rapid_switch_seconds,
):
    with pytest.raises(ValueError, match="vllm_rapid_switch_seconds"):
        PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
            vllm_rapid_switch_seconds=rapid_switch_seconds,
        ).validate()


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("request_timeout", 0),
        ("request_timeout", float("inf")),
        ("request_timeout", True),
        ("request_attempt_timeout", -1),
        ("request_attempt_timeout", float("nan")),
        ("request_attempt_timeout", "300"),
    ],
)
def test_stage2_config_rejects_invalid_request_timeouts(field_name, value):
    with pytest.raises(ValueError, match=field_name):
        PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
            **{field_name: value},
        ).validate()


def test_stage2_config_rejects_invalid_consolidation_iteration_limits():
    with pytest.raises(ValueError, match="operationalization_max_prompt_chars"):
        PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
            operationalization_max_prompt_chars=3_999,
        ).validate()

    with pytest.raises(ValueError, match="consolidation_batch_size must be at least 2"):
        PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
            consolidation_batch_size=1,
        ).validate()

    with pytest.raises(ValueError, match="consolidation_max_rounds must be positive"):
        PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
            consolidation_max_rounds=0,
        ).validate()

    with pytest.raises(ValueError, match="consolidation_alphabetical_rounds"):
        PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
            consolidation_alphabetical_rounds=-1,
        ).validate()

    with pytest.raises(ValueError, match="ontology_refinement_min_failure_patients"):
        PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
            ontology_refinement_min_failure_patients=1,
        ).validate()

    with pytest.raises(ValueError, match="max_ontology_refinement_rounds"):
        PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
            max_ontology_refinement_rounds=-1,
        ).validate()

    with pytest.raises(ValueError, match="extraction_feature_batch_size"):
        PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
            extraction_feature_batch_size=0,
        ).validate()

    with pytest.raises(ValueError, match="extraction_chunk_size_tokens"):
        PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
            extraction_chunk_size_tokens=0,
        ).validate()

    with pytest.raises(ValueError, match="context window must exceed"):
        PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
            extraction_context_window_tokens=76_024,
        ).validate()


@pytest.mark.parametrize("max_tokens", [0, -1, 99_999, True, 1.5, "100000"])
def test_stage2_config_rejects_invalid_max_tokens(max_tokens):
    with pytest.raises(ValueError, match="max_tokens must be an integer of at least 100000"):
        PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
            max_tokens=max_tokens,
        ).validate()


@pytest.mark.parametrize(
    "extraction_max_tokens",
    [0, -1, 4_095, True, 1.5, "4096"],
)
def test_stage2_config_rejects_invalid_extraction_max_tokens(extraction_max_tokens):
    with pytest.raises(
        ValueError,
        match="extraction_max_tokens must be an integer of at least 4096",
    ):
        PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
            extraction_max_tokens=extraction_max_tokens,
        ).validate()


@pytest.mark.parametrize("max_response_repairs", [-1, True, 1.5, "10"])
def test_stage2_config_rejects_invalid_max_response_repairs(max_response_repairs):
    with pytest.raises(ValueError, match="max_response_repairs must be a nonnegative integer"):
        PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
            max_response_repairs=max_response_repairs,
        ).validate()


@pytest.mark.parametrize("thinking_after_response_repairs", [-1, True, 1.5, "5"])
def test_stage2_config_rejects_invalid_thinking_repair_threshold(
    thinking_after_response_repairs,
):
    with pytest.raises(
        ValueError,
        match="thinking_after_response_repairs must be a nonnegative integer",
    ):
        PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
            thinking_after_response_repairs=thinking_after_response_repairs,
        ).validate()


@pytest.mark.parametrize(
    "repetition_penalty",
    [0, -1, True, float("inf"), float("nan"), "1.1"],
)
def test_stage2_config_rejects_invalid_repetition_penalty(repetition_penalty):
    with pytest.raises(ValueError, match="repetition_penalty"):
        PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
            repetition_penalty=repetition_penalty,
        ).validate()


@pytest.mark.parametrize(
    "field_name",
    ["interpretation_reasoning_effort", "extraction_reasoning_effort"],
)
def test_stage2_config_rejects_invalid_reasoning_effort(field_name):
    with pytest.raises(ValueError, match=field_name):
        PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
            **{field_name: "very-hard"},
        ).validate()


def test_stage2_config_parses_explicit_feature_with_supplied_ontology():
    config = plain_stage2_config_from_mapping(
        {
            "endpoint": "http://stage2.test/v1",
            "explicit_features": [
                {
                    "name": "ECOG performance status",
                    "roles": ["confounder", "effect_modifier"],
                    "ontology": {
                        "description": "Pretreatment ECOG performance status.",
                        "value_type": "ordinal",
                        "categories_or_unit": ["0-4"],
                        "measurement_definition": (
                            "Extract the last explicitly documented pretreatment ECOG score."
                        ),
                        "missing_value_rule": "Return null when no score is documented.",
                    },
                }
            ],
        },
        default_workers=1,
    )

    assert config is not None
    assert len(config.explicit_features) == 1
    feature = config.explicit_features[0]
    assert feature.name == "ecog_performance_status"
    assert feature.value_type == "ordinal"
    assert feature.categories_or_unit == ("0", "1", "2", "3", "4")
    assert feature.roles == ("confounder", "effect_modifier")
    assert feature.conflict_resolution == {
        "strategy": "latest",
        "positive_category": None,
    }
    assert config.public_dict()["explicit_features"][0]["categories_or_unit"] == [
        "0",
        "1",
        "2",
        "3",
        "4",
    ]


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


def test_stage2_rejects_retired_raw_packet_compiler():
    with pytest.raises(ValueError, match="raw_packets_v1 was retired"):
        plain_stage2_config_from_mapping(
            {
                "endpoint": "http://stage2.test/v1",
                "evidence_compiler": "raw_packets_v1",
            },
            default_workers=1,
        )


def test_stage2_ignores_legacy_extraction_batch_size_and_parses_extraction_token_cap(caplog):
    config = plain_stage2_config_from_mapping(
        {
            "endpoint": "http://stage2.test/v1",
            "extraction_batch_size": 100,
            "max_tokens": 125_000,
        },
        default_workers=8,
    )

    assert config is not None
    assert "extraction_batch_size" not in config.public_dict()
    assert config.max_tokens == 125_000
    assert config.public_dict()["max_tokens"] == 125_000
    assert "permanently isolated to one patient per prompt" in caplog.text


def test_stage2_maps_legacy_enable_thinking_to_interpretation_effort(caplog):
    config = plain_stage2_config_from_mapping(
        {
            "endpoint": "http://stage2.test/v1",
            "enable_thinking": False,
        },
        default_workers=1,
    )

    assert config is not None
    assert config.interpretation_reasoning_effort == "none"
    assert config.extraction_reasoning_effort == "auto"
    assert "enable_thinking is deprecated" in caplog.text


def test_stage2_autodiscovers_the_only_served_model(monkeypatch):
    monkeypatch.setattr(
        stage2_workflow,
        "_served_model_ids",
        lambda _config: ["served-model"],
    )

    runner = PlainHandoffStage2(
        config=PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
        ),
        clinical_question="Identify confounders.",
        completion=lambda _messages, _config: "{}",
    )

    assert runner.config.model == "served-model"


def test_stage2_autodiscovery_rejects_multiple_served_models(monkeypatch):
    monkeypatch.setattr(
        stage2_workflow,
        "_served_model_ids",
        lambda _config: ["model-a", "model-b"],
    )

    with pytest.raises(RuntimeError, match="requires exactly one served model"):
        PlainHandoffStage2(
            config=PlainHandoffStage2Config(
                endpoint="http://stage2.test/v1",
            ),
            clinical_question="Identify confounders.",
            completion=lambda _messages, _config: "{}",
        )


def test_explicit_stage2_model_skips_autodiscovery(monkeypatch):
    def unexpected_discovery(_config):
        raise AssertionError("model discovery should not run")

    monkeypatch.setattr(stage2_workflow, "_served_model_ids", unexpected_discovery)

    runner = PlainHandoffStage2(
        config=PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="configured-model",
        ),
        clinical_question="Identify confounders.",
        completion=lambda _messages, _config: "{}",
    )

    assert runner.config.model == "configured-model"


def test_live_stage2_verifies_the_configured_model_on_every_runtime_endpoint(
    monkeypatch,
):
    checked = []

    def served_models(config):
        checked.append(config.endpoint)
        return ["configured-model"]

    monkeypatch.setattr(stage2_workflow, "_served_model_ids", served_models)

    runner = PlainHandoffStage2(
        config=PlainHandoffStage2Config(
            endpoint="http://replica-a.test/v1",
            runtime_endpoints=(
                "http://replica-a.test/v1",
                "http://replica-b.test/v1",
            ),
            model="configured-model",
        ),
        clinical_question="Identify confounders.",
    )

    assert checked == ["http://replica-a.test/v1", "http://replica-b.test/v1"]
    assert runner.model_identity["primary"]["selected_model"] == "configured-model"
    assert runner.model_identity["primary"]["live_endpoint_verified"] is True


def test_live_stage2_rejects_configured_model_that_endpoint_does_not_serve(
    monkeypatch,
):
    monkeypatch.setattr(
        stage2_workflow,
        "_served_model_ids",
        lambda _config: ["different-running-model"],
    )

    with pytest.raises(RuntimeError, match="actual served model does not match"):
        PlainHandoffStage2(
            config=PlainHandoffStage2Config(
                endpoint="http://stage2.test/v1",
                model="configured-model",
            ),
            clinical_question="Identify confounders.",
        )


def test_same_live_endpoint_can_serve_distinct_primary_and_extraction_models(
    monkeypatch,
):
    monkeypatch.setattr(
        stage2_workflow,
        "_served_model_ids",
        lambda _config: ["large-reviewer", "small-extractor"],
    )

    runner = PlainHandoffStage2(
        config=PlainHandoffStage2Config(
            endpoint="http://shared.test/v1",
            model="large-reviewer",
            extraction_llm=Stage2ExtractionLLMConfig(
                endpoint="http://shared.test/v1",
                model="small-extractor",
            ),
        ),
        clinical_question="Identify confounders.",
    )

    assert runner.model_identity["primary"]["selected_model"] == "large-reviewer"
    assert runner.model_identity["extraction"]["selected_model"] == "small-extractor"


@pytest.mark.parametrize(
    ("model", "family"),
    [
        ("Qwen/Qwen3.8-27B", "qwen3"),
        ("google/gemma-4-26B-A4B-it", "gemma4"),
        ("LiquidAI/LFM2.5-1.2B-Thinking", "lfm2.5"),
        ("some/other-model", "other"),
    ],
)
def test_stage2_detects_reasoning_model_family_from_served_model_id(model, family):
    assert stage2_workflow._stage2_model_family(model) == family


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


def test_model_identity_resume_allows_extractor_change_before_extraction(
    tmp_path: Path,
):
    completion = lambda _messages, _config: "{}"

    def runner(extraction_model):
        return PlainHandoffStage2(
            config=PlainHandoffStage2Config(
                endpoint="http://stage2.test/v1",
                model="same-primary",
                extraction_llm=Stage2ExtractionLLMConfig(
                    endpoint="http://extract.test/v1",
                    model=extraction_model,
                ),
            ),
            clinical_question="Identify confounders.",
            completion=completion,
        )

    runner("extractor-a")._check_and_record_model_identity(tmp_path)
    runner("extractor-b")._check_and_record_model_identity(tmp_path)

    identity = json.loads((tmp_path / "model_identity.json").read_text(encoding="utf-8"))
    assert identity["extraction"]["selected_model"] == "extractor-b"


def test_model_identity_resume_rejects_extractor_change_with_extraction_checkpoints(
    tmp_path: Path,
):
    completion = lambda _messages, _config: "{}"

    def runner(extraction_model):
        return PlainHandoffStage2(
            config=PlainHandoffStage2Config(
                endpoint="http://stage2.test/v1",
                model="same-primary",
                extraction_llm=Stage2ExtractionLLMConfig(
                    endpoint="http://extract.test/v1",
                    model=extraction_model,
                ),
            ),
            clinical_question="Identify confounders.",
            completion=completion,
        )

    runner("extractor-a")._check_and_record_model_identity(tmp_path)
    (tmp_path / "outer_001" / "ontology_supervision").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="extraction-dependent checkpoints remain"):
        runner("extractor-b")._check_and_record_model_identity(tmp_path)


@pytest.mark.parametrize("role", ["primary", "extraction"])
@pytest.mark.parametrize("field", ["root", "parent", "revision"])
def test_model_identity_resume_allows_backing_change_behind_same_served_alias(
    tmp_path: Path,
    monkeypatch,
    role,
    field,
):
    backing_record = {
        "id": "gemma4-31b",
        "root": "RedHatAI/Gemma-4-31B-IT-FP8-Dynamic",
        "parent": None,
        "revision": None,
    }

    def served_models(_config):
        return stage2_workflow._ServedModelIds(
            ["gemma4-31b"],
            records=[dict(backing_record)],
        )

    monkeypatch.setattr(stage2_workflow, "_served_model_ids", served_models)

    def runner():
        return PlainHandoffStage2(
            config=PlainHandoffStage2Config(
                endpoint="http://stage2.test/v1",
                model="gemma4-31b",
                extraction_llm=(
                    Stage2ExtractionLLMConfig(
                        endpoint="http://extract.test/v1",
                        model="gemma4-31b",
                    )
                    if role == "extraction"
                    else None
                ),
            ),
            clinical_question="Identify confounders.",
        )

    first = runner()
    assert first.config.runtime_model_family == "gemma4"
    first._check_and_record_model_identity(tmp_path)
    # Existing extraction checkpoints must also allow a stable served name.
    (tmp_path / "outer_001" / "ontology_supervision").mkdir(parents=True)

    backing_record[field] = "nvidia/Gemma-4-31B-IT-NVFP4"
    changed = runner()
    changed._check_and_record_model_identity(tmp_path)

    identity = json.loads((tmp_path / "model_identity.json").read_text(encoding="utf-8"))
    assert identity[role]["selected_model"] == "gemma4-31b"
    assert identity[role]["actual_model_identity"][field] == backing_record[field]


def test_json_repair_retry_stays_within_full_initial_prompt_budget():
    conversations = []

    def completion(messages, _config):
        conversations.append([dict(message) for message in messages])
        response_attempt = len(conversations)
        return json.dumps(
            {
                "ok": response_attempt == 6,
                "response_attempt": response_attempt,
            }
        )

    def validate(value):
        if value["ok"] is not True:
            raise ValueError(f"response attempt {value['response_attempt']} rejected")
        return dict(value)

    config = PlainHandoffStage2Config(
        endpoint="http://stage2.test/v1",
        model="test-model",
        max_prompt_chars=4_000,
    )
    result = stage2_workflow._request_json(
        messages=[
            {"role": "system", "content": "S" * 100},
            {"role": "user", "content": "U" * 3_900},
        ],
        config=config,
        completion=completion,
        validate=validate,
    )

    assert result == {"ok": True, "response_attempt": 6}
    assert len(conversations) == 6
    assert all(
        sum(len(message["content"]) for message in conversation) <= 4_000
        for conversation in conversations
    )
    for repair_attempt in range(1, 6):
        assert (
            f"response attempt {repair_attempt} rejected"
            in conversations[repair_attempt][0]["content"]
        )


def test_json_repair_stops_after_fifteen_repairs():
    calls = []

    def completion(messages, _config):
        calls.append([dict(message) for message in messages])
        return "{}"

    def reject(_value):
        raise ValueError("missing ok=true")

    with pytest.raises(ValueError, match="remained invalid after 15 repairs"):
        stage2_workflow._request_json(
            messages=[
                {"role": "system", "content": "Return JSON only."},
                {"role": "user", "content": "Return an object containing ok=true."},
            ],
            config=PlainHandoffStage2Config(
                endpoint="http://stage2.test/v1",
                model="test-model",
            ),
            completion=completion,
            validate=reject,
        )

    assert len(calls) == 16


def test_json_repair_audits_invalid_responses_and_uses_repeated_error_fallback():
    conversations = []
    events = []

    def completion(messages, _config):
        conversations.append([dict(message) for message in messages])
        return '{"ok":false,"condition":{"operator":"present"}}'

    def validate(value):
        if value.get("ok") is not True:
            raise ValueError("rule condition references unavailable feature ''")
        return dict(value)

    fallback = {
        "ok": True,
        "action": "leave_unchanged",
        "rationale": "Conservative fallback.",
    }
    result = stage2_workflow._request_json(
        messages=[
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": "Return a schema-valid decision."},
        ],
        config=PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
        ),
        completion=completion,
        validate=validate,
        repair_context={
            "allowed_feature_ids": ["feature_a", "feature_b"],
            "valid_expression_examples": {
                "coalesce": {
                    "op": "coalesce",
                    "feature_ids": ["feature_a", "feature_b"],
                }
            },
            "conservative_response": fallback,
        },
        validation_event_observer=lambda event: events.append(dict(event)),
        conservative_validation_fallback=fallback,
        fallback_after_same_error=3,
    )

    assert result == fallback
    assert len(conversations) == 3
    assert "rule condition references unavailable feature" in conversations[1][-1]["content"]
    assert "complete corrected JSON object" in conversations[2][-1]["content"]
    assert "Valid categorical-rule expression examples" not in conversations[1][-1]["content"]
    invalid_events = [event for event in events if event["event"] == "invalid_response"]
    assert len(invalid_events) == 3
    assert invalid_events[0]["raw_response"].startswith('{"ok":false')
    assert invalid_events[-1]["same_error_occurrence"] == 3
    assert events[-1]["event"] == "conservative_fallback"
    assert events[-1]["trigger"] == "repeated_identical_validation_error"


def test_extraction_repairs_dynamically_cap_output_to_the_remaining_context():
    attempts = []

    def counter(messages):
        return 10 + sum(len(message["content"]) for message in messages)

    def completion(messages, request_config):
        prompt_tokens = counter(messages)
        output_cap = stage2_workflow._stage2_request_policy(request_config)["max_tokens"]
        attempts.append((prompt_tokens, output_cap))
        return "{}" if len(attempts) == 1 else '{"ok": true}'

    result = stage2_workflow._request_json(
        messages=[
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": "x" * 300},
        ],
        config=PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
            extraction_max_tokens=75_000,
            max_prompt_chars=4_000,
        ),
        completion=completion,
        validate=lambda value: (
            dict(value)
            if value.get("ok") is True
            else (_ for _ in ()).throw(ValueError("missing ok=true"))
        ),
        request_kind="extraction",
        prompt_token_counter=counter,
        context_window_tokens=1_000,
        context_margin_tokens=100,
    )

    assert result == {"ok": True}
    assert len(attempts) == 2
    assert all(prompt + output + 100 <= 1_000 for prompt, output in attempts)
    assert all(output < 75_000 for _prompt, output in attempts)


def test_output_length_repair_explicitly_requests_a_shorter_complete_object():
    message = stage2_workflow._repair_message(
        stage2_workflow._Stage2OutputLengthError(
            "Stage 2 server stopped the response with finish_reason=length"
        )
    )

    assert message["role"] == "user"
    assert "complete JSON object" in message["content"]
    assert "Keep explanations concise" in message["content"]
    assert "Include every requested entry and field" in message["content"]


def test_json_repair_losslessly_compacts_json_to_include_the_validation_error():
    conversations = []
    payload = {"items": [f"item-{index}" for index in range(500)]}
    rendered = json.dumps(payload, sort_keys=True)

    def completion(messages, _config):
        conversations.append([dict(message) for message in messages])
        return "{}" if len(conversations) == 1 else '{"ok": true}'

    def validate(value):
        if value.get("ok") is not True:
            raise ValueError("missing required ok field")
        return dict(value)

    config = PlainHandoffStage2Config(
        endpoint="http://stage2.test/v1",
        model="test-model",
        max_prompt_chars=len(rendered) + 100,
    )
    result = stage2_workflow._request_json(
        messages=[
            {"role": "system", "content": "S" * 100},
            {"role": "user", "content": rendered},
        ],
        config=config,
        completion=completion,
        validate=validate,
    )

    assert result == {"ok": True}
    assert len(conversations[1]) == 4
    assert conversations[1][2] == {"role": "assistant", "content": "{}"}
    assert json.loads(conversations[1][1]["content"]) == payload
    assert "missing required ok field" in conversations[1][3]["content"]
    assert sum(len(message["content"]) for message in conversations[1]) <= (config.max_prompt_chars)


def test_extraction_category_error_lists_allowed_literals_and_prompts_forbid_aliases():
    definition = {
        "feature_id": "outer_001_feature_001",
        "name": "prior_immunotherapy_history",
        "description": "Whether prior immunotherapy was documented.",
        "value_type": "binary",
        "categories_or_unit": ["not documented", "documented"],
        "roles": ["confounder"],
        "measurement_definition": "Extract documented immunotherapy history.",
        "missing_value_rule": "Return null when the history is unavailable.",
        "supporting_packet_ids": ["packet_001"],
        "supporting_architectures": ["test_architecture"],
        "stability_summary": "Supported by several discovery contexts.",
        "caveats": "Documentation may be incomplete.",
    }
    with pytest.raises(ValueError) as error:
        stage2_analysis._validate_extraction(
            {
                "rows": [
                    {
                        "row_id": 7,
                        "values": {"prior_immunotherapy_history": 1},
                    }
                ]
            },
            row_ids=[7],
            definitions=[definition],
        )

    message = str(error.value)
    assert "value 1 is invalid" in message
    assert '["not documented","documented"] or null' in message

    extraction = stage2_analysis._extraction_prompt(definitions=[definition], rows=[{"row_id": 7, "text": "Prior immunotherapy was documented."}])
    page = stage2_analysis._page_extraction_prompt(definitions=[definition], row={"row_id": 7, "text": "Prior immunotherapy was documented."})
    for messages in (extraction, page):
        assert "Allowed categories: not documented; documented" in messages[1]["content"]
        assert "row_id" not in str(messages) and "record_scope" not in str(messages)
    assert "Use the defined category labels exactly" in extraction[0]["content"]
    assert "Preserve repeated or conflicting observations" in page[0]["content"]


def test_normal_extraction_prompt_applies_explicit_conflict_resolution():
    definition = {
        "name": "age",
        "description": "The patient's age in years.",
        "value_type": "continuous",
        "categories_or_unit": ["years"],
        "measurement_definition": (
            "Extract the age from the pretreatment record or earliest available encounter."
        ),
        "missing_value_rule": "Return null when age is undocumented.",
        "conflict_resolution": {"strategy": "latest", "positive_category": None},
    }

    messages = stage2_analysis._extraction_prompt(definitions=[definition], rows=[{"row_id": 7, "text": "Age 68 at diagnosis; age 72 today."}])
    assert "Choose among repeated observations using this rule: Use the latest dated observation" in messages[1]["content"]
    assert "use the last mention" in messages[1]["content"]
    assert "row_id" not in str(messages)


def test_continuous_extraction_preserves_categorical_fallback_for_modeling_review():
    definition = {
        "feature_id": "outer_001_feature_001",
        "name": "pd_l1_tumor_proportion_score",
        "description": "PD-L1 tumor proportion score.",
        "value_type": "continuous",
        "categories_or_unit": ["percent"],
        "roles": ["effect_modifier"],
        "measurement_definition": (
            "Extract the numeric TPS when present, otherwise preserve its documented category."
        ),
        "missing_value_rule": "Return null when TPS is undocumented.",
    }
    validated = stage2_analysis._validate_extraction(
        {
            "rows": [
                {
                    "row_id": 7,
                    "values": {"pd_l1_tumor_proportion_score": "<1%"},
                }
            ]
        },
        row_ids=[7],
        definitions=[definition],
    )
    prompt = stage2_analysis._extraction_prompt(definitions=[definition], rows=[{"row_id": 7, "text": "PD-L1 TPS was reported as <1%."}])[1]["content"]
    frame = pd.DataFrame(
        {
            "_oci_row_id": [0, 1, 2, 3],
            "pd_l1_tumor_proportion_score": [50.0, "<1%", 20.0, None],
        }
    )
    summary = stage2_analysis.feature_summaries(frame, [definition])[0]
    hybrid = {
        **definition,
        "modeling_strategy": "continuous_with_categorical_fallback",
    }
    encoded = stage2_analysis._FeatureEncoder([hybrid]).fit(frame).transform(frame)

    assert validated["rows"][0]["values"]["pd_l1_tumor_proportion_score"] == "<1%"
    assert "categorical/threshold string" in prompt
    assert summary["numeric_nonmissing"] == 2
    assert summary["categorical_fallback_nonmissing"] == 1
    assert summary["categorical_fallback_values"] == {"<1%": 1}
    assert summary["recommended_modeling_strategy"] == ("continuous_with_categorical_fallback")
    assert encoded.shape == (4, 5)
    assert np.isfinite(encoded).all()
    assert encoded[1, 3] == 1.0


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


def test_stage2_extraction_batches_features_by_default_and_accepts_override(
    tmp_path: Path,
):
    definitions = [
        {
            "name": f"feature_{index:02d}",
            "description": f"Feature {index}",
            "value_type": "continuous",
            "categories_or_unit": ["score"],
            "measurement_definition": f"Extract feature {index}.",
            "missing_value_rule": "Return null when undocumented.",
        }
        for index in range(23)
    ]
    dataset = pd.DataFrame({"clinical_text": ["All feature values are documented."]})

    def run_extraction(output_dir: Path, **kwargs):
        prompt_feature_names = []

        def request_json(messages, validate, *, request_kind="interpretation"):
            body = prompt_inputs(messages)
            names = [feature["name"] for feature in body["features"]]
            prompt_feature_names.append(names)
            assert len(body["patients"]) == 1
            return validate(
                {
                    "rows": [
                        {
                            "row_id": 0,
                            "values": {name: int(name.removeprefix("feature_")) for name in names},
                        }
                    ]
                }
            )

        frame = extract_rows(
            dataset=dataset,
            row_ids=[0],
            text_column="clinical_text",
            definitions=definitions,
            output_dir=output_dir,
            request_json=request_json,
            workers=1,
            max_prompt_chars=100_000,
            **kwargs,
        )
        return frame, prompt_feature_names

    default_output = tmp_path / "default"
    frame, default_prompts = run_extraction(default_output)

    assert [len(names) for names in default_prompts] == [10, 10, 3]
    assert [name for names in default_prompts for name in names] == [
        definition["name"] for definition in definitions
    ]
    assert frame.loc[0, "feature_00"] == 0.0
    assert frame.loc[0, "feature_22"] == 22.0
    parent_completion = json.loads(
        (default_output / "batches" / "batch_00001" / "complete.json").read_text(encoding="utf-8")
    )
    assert parent_completion["feature_batches"] == 3
    assert parent_completion["feature_batch_size"] == 10
    assert (
        len(
            list(
                (default_output / "batches" / "batch_00001" / "feature_batches").glob(
                    "batch_*/complete.json"
                )
            )
        )
        == 3
    )

    configured_frame, configured_prompts = run_extraction(
        tmp_path / "configured",
        feature_batch_size=6,
    )
    assert [len(names) for names in configured_prompts] == [6, 6, 6, 5]
    assert configured_frame.loc[0, "feature_22"] == 22.0

    def unexpected_request(_messages, _validate, *, request_kind="interpretation"):
        raise AssertionError("feature-batch checkpoints should be reused")

    parent_dir = default_output / "batches" / "batch_00001"
    (parent_dir / "complete.json").unlink()
    (parent_dir / "result.json").unlink()
    rebuilt = extract_rows(
        dataset=dataset,
        row_ids=[0],
        text_column="clinical_text",
        definitions=definitions,
        output_dir=default_output,
        request_json=unexpected_request,
        workers=1,
        max_prompt_chars=100_000,
    )
    pd.testing.assert_frame_equal(rebuilt, frame)

    resumed = extract_rows(
        dataset=dataset,
        row_ids=[0],
        text_column="clinical_text",
        definitions=definitions,
        output_dir=default_output,
        request_json=unexpected_request,
        workers=1,
        max_prompt_chars=100_000,
    )
    pd.testing.assert_frame_equal(resumed, frame)


def test_extraction_retries_only_legacy_infrastructure_failure_feature_batches(
    tmp_path: Path,
):
    definitions = [
        {
            "name": f"feature_{index:02d}",
            "description": f"Feature {index}",
            "value_type": "continuous",
            "categories_or_unit": ["score"],
            "measurement_definition": f"Extract feature {index}.",
            "missing_value_rule": "Return null when undocumented.",
        }
        for index in range(2)
    ]
    dataset = pd.DataFrame({"clinical_text": ["Both feature values are documented."]})
    output = tmp_path / "extraction"
    calls: list[list[str]] = []

    def request_json(messages, validate, *, request_kind="interpretation"):
        assert request_kind == "extraction"
        body = prompt_inputs(messages)
        names = [feature["name"] for feature in body["features"]]
        calls.append(names)
        return validate(
            {
                "rows": [
                    {
                        "row_id": 0,
                        "values": {
                            name: int(name.removeprefix("feature_")) for name in names
                        },
                    }
                ]
            }
        )

    expected = extract_rows(
        dataset=dataset,
        row_ids=[0],
        text_column="clinical_text",
        definitions=definitions,
        output_dir=output,
        request_json=request_json,
        workers=1,
        max_prompt_chars=100_000,
        feature_batch_size=1,
    )
    calls.clear()

    parent_dir = output / "batches" / "batch_00001"
    failed_dir = parent_dir / "feature_batches" / "batch_00002"
    (failed_dir / "result.json").write_text(
        json.dumps({"rows": [{"row_id": 0, "values": {"feature_01": None}}]}),
        encoding="utf-8",
    )
    (failed_dir / "extraction_failure.json").write_text(
        json.dumps(
            {
                "resolution": "conservative_all_null",
                "validation_error": (
                    "Stage2RequestExhaustedError: transport attempts exhausted"
                ),
            }
        ),
        encoding="utf-8",
    )
    (failed_dir / "extraction_issues.json").write_text(
        json.dumps(
            {
                "events": [
                    {
                        "failure_kind": "structural_response_failure",
                        "reason": "Stage2RequestExhaustedError: transport attempts exhausted",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (parent_dir / "result.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "row_id": 0,
                        "values": {"feature_00": 0, "feature_01": None},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    repaired = extract_rows(
        dataset=dataset,
        row_ids=[0],
        text_column="clinical_text",
        definitions=definitions,
        output_dir=output,
        request_json=request_json,
        workers=1,
        max_prompt_chars=100_000,
        feature_batch_size=1,
    )

    pd.testing.assert_frame_equal(repaired, expected)
    assert calls == [["feature_01"]]
    assert (failed_dir / "superseded_infrastructure_extraction_failure.json").is_file()
    assert (failed_dir / "superseded_infrastructure_result.json").is_file()
    assert (parent_dir / "superseded_infrastructure_complete.json").is_file()
    archived_failure = output / "_partial_backup_20260827" / "extraction_failure.json"
    archived_failure.parent.mkdir()
    archived_failure.write_text(
        '{"validation_error":"Stage2RequestExhaustedError: archived"}',
        encoding="utf-8",
    )
    assert stage2_analysis.infrastructure_failure_audit_paths(output) == ()


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


def test_serial_ontology_repair_normalizes_numeric_carry_forward_state(
    tmp_path: Path,
):
    definitions = [
        {
            "feature_id": "outer_001_feature_207",
            "name": "metastasis_sites",
            "description": "Documented sites of distant metastasis.",
            "value_type": "categorical",
            "categories_or_unit": ["Brain", "Bone", "Liver", "Adrenal Gland"],
            "measurement_definition": "Extract one declared metastatic site.",
            "missing_value_rule": "Return null when unavailable.",
            "conflict_resolution": {"strategy": "latest", "positive_category": None},
        },
        {
            "feature_id": "outer_001_feature_208",
            "name": "metastasis_size",
            "description": "Documented metastatic lesion size.",
            "value_type": "continuous",
            "categories_or_unit": ["cm"],
            "measurement_definition": "Extract the documented lesion size.",
            "missing_value_rule": "Return null when unavailable.",
            "conflict_resolution": {"strategy": "latest", "positive_category": None},
        },
    ]
    note = "metastatic disease " + ("x" * 450)
    observed_prior_states = []
    extraction_calls = 0

    def request_json(messages, validate, *, request_kind="interpretation"):
        nonlocal extraction_calls
        body = prompt_inputs(messages)
        if request_kind == "interpretation":
            item = body["items"][0]
            return validate(
                {
                    "corrections": [
                        {"mapping_id": item["mapping_id"], "value": None}
                    ]
                }
            )

        extraction_calls += 1
        patient = body["patient"]
        observed_prior_states.append(dict(patient["prior_feature_state"]))
        if extraction_calls == 1:
            values = {
                "metastasis_sites": "Adrenal Gland, Bone",
                "metastasis_size": 4.5,
            }
            state = {
                "metastasis_sites": "Adrenal Gland and bone are documented.",
                # This is the exact malformed metadata shape from the failed run.
                "metastasis_size": 4.5,
            }
        else:
            values = dict(patient["prior_extraction"])
            state = dict(patient["prior_feature_state"])
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

    output = tmp_path / "serial_ontology_repair"
    frame = extract_rows(
        dataset=pd.DataFrame({"clinical_text": [note]}),
        row_ids=[0],
        text_column="clinical_text",
        definitions=definitions,
        output_dir=output,
        request_json=request_json,
        workers=1,
        max_prompt_chars=100_000,
        tokenizer=_CharacterChatTokenizer(),
        chunk_size_tokens=200,
        context_window_tokens=6_000,
        max_output_tokens=500,
        context_margin_tokens=200,
    )

    assert extraction_calls > 1
    assert pd.isna(frame.loc[0, "metastasis_sites"])
    assert frame.loc[0, "metastasis_size"] == 4.5
    assert observed_prior_states[1]["metastasis_size"] == "4.5"
    first_chunk = (
        output
        / "batches"
        / "batch_00001"
        / "serial_chunks"
        / "chunk_00001"
    )
    audit = json.loads(
        (first_chunk / "category_ontology_repair.json").read_text(encoding="utf-8")
    )
    assert audit["resolution"] == "llm_category_ontology"
    assert (output / "complete.json").is_file()


def test_stage2_tokenizer_keeps_short_patient_extraction_one_shot(tmp_path: Path):
    jobs = []

    def request_json(messages, validate, *, request_kind="interpretation"):
        body = prompt_inputs(messages)
        jobs.append(body["job"])
        patient = body["patients"][0]
        return validate(
            {
                "rows": [
                    {
                        "row_id": patient["row_id"],
                        "values": {"age": 67},
                    }
                ]
            }
        )

    frame = extract_rows(
        dataset=pd.DataFrame({"clinical_text": ["Age 67."]}),
        row_ids=[0],
        text_column="clinical_text",
        definitions=[
            {
                "name": "age",
                "value_type": "continuous",
                "categories_or_unit": ["years"],
            }
        ],
        output_dir=tmp_path / "short",
        request_json=request_json,
        workers=1,
        max_prompt_chars=100_000,
        tokenizer=_CharacterChatTokenizer(),
        chunk_size_tokens=50_000,
        context_window_tokens=10_000,
        max_output_tokens=1_000,
        context_margin_tokens=200,
    )

    assert jobs == ["extract_stage2_patient_variables"]
    assert frame.loc[0, "age"] == 67.0
    assert not list((tmp_path / "short").glob("**/serial_complete.json"))


def test_stage2_extraction_prompt_does_not_ascii_escape_clinical_text():
    note = "患者は治療前に息切れを報告した。"
    definition = {
        "name": "dyspnea",
        "value_type": "binary",
        "categories_or_unit": ["present", "absent"],
    }

    messages = stage2_analysis._extraction_prompt(
        definitions=[definition],
        rows=[{"row_id": 0, "text": note}],
    )

    assert note in messages[1]["content"]
    assert "\\u60a3" not in messages[1]["content"]
    assert messages[1]["content"].endswith(note)


def test_ordinal_integer_range_is_expanded_in_prompt_and_validation():
    definition = {
        "name": "performance_status",
        "value_type": "ordinal",
        "categories_or_unit": ["0-4"],
    }
    prompt = stage2_analysis._extraction_prompt(definitions=[definition], rows=[{"row_id": 3, "text": "ECOG was 2."}])[1]["content"]
    validated = stage2_analysis._validate_extraction(
        {
            "rows": [
                {
                    "row_id": 3,
                    "values": {"performance_status": 2},
                }
            ]
        },
        row_ids=[3],
        definitions=[definition],
    )

    assert "Allowed categories: 0; 1; 2; 3; 4" in prompt
    assert validated["rows"][0]["values"]["performance_status"] == "2"
    assert stage2_analysis._normalized_category_values(
        value_type="categorical",
        values=["0-4"],
    ) == ["0", "1", "2", "3", "4"]
    assert stage2_analysis._normalized_category_values(
        value_type="categorical",
        values=["0-4", "5-9"],
    ) == ["0-4", "5-9"]
    assert stage2_analysis._normalized_category_values(
        value_type="binary",
        values=["presence/absence"],
    ) == ["presence", "absence"]


def test_extraction_recovers_feature_key_drift_and_defaults_missing_values_to_null(caplog):
    definitions = [
        {
            "name": "prior_immunotherapy_history",
            "value_type": "binary",
            "categories_or_unit": ["not documented", "documented"],
        },
        {
            "name": "performance_status",
            "value_type": "ordinal",
            "categories_or_unit": ["0-4"],
        },
    ]
    validated = stage2_analysis._validate_extraction(
        {
            "rows": [
                {
                    "row_id": 9,
                    "values": {
                        "Prior Immunotherapy History": "documented",
                        "invented_feature": "ignore me",
                    },
                }
            ]
        },
        row_ids=[9],
        definitions=definitions,
    )

    assert validated == {
        "rows": [
            {
                "row_id": 9,
                "values": {
                    "prior_immunotherapy_history": "documented",
                    "performance_status": None,
                },
            }
        ]
    }
    assert "missing_as_null=['performance_status']" in caplog.text
    assert "extras_dropped=['invented_feature']" in caplog.text


def test_resume_adopts_valid_legacy_single_patient_extraction_checkpoint(
    tmp_path: Path,
    caplog,
):
    caplog.set_level("INFO", logger=stage2_analysis.__name__)
    output = tmp_path / "extraction"
    batch = output / "batches" / "batch_00001"
    batch.mkdir(parents=True)
    (batch / "row_ids.json").write_text("[0]\n", encoding="utf-8")
    (batch / "result.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "row_id": 0,
                        "values": {"performance_status": "2"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (batch / "complete.json").write_text(
        json.dumps({"status": "complete", "rows": 1}),
        encoding="utf-8",
    )
    definition = {
        "name": "performance_status",
        "value_type": "ordinal",
        "categories_or_unit": ["0-4"],
    }

    def unexpected_request(_messages, _validate, *, request_kind="interpretation"):
        raise AssertionError("valid legacy singleton checkpoint should be adopted")

    frame = extract_rows(
        dataset=pd.DataFrame({"clinical_text": ["Pretreatment ECOG was 2."]}),
        row_ids=[0],
        text_column="clinical_text",
        definitions=[definition],
        output_dir=output,
        request_json=unexpected_request,
        workers=1,
        max_prompt_chars=10_000,
    )

    assert frame.loc[0, "performance_status"] == "2"
    completion = json.loads((batch / "complete.json").read_text(encoding="utf-8"))
    assert completion["schema_version"] == stage2_analysis.EXTRACTION_CHECKPOINT_SCHEMA_VERSION
    assert completion["adopted_legacy_single_patient_checkpoint"] is True
    assert "adopt legacy single-patient" in caplog.text


def test_resume_relocates_legacy_singleton_by_row_id_after_batch_numbers_shift(
    tmp_path: Path,
    caplog,
):
    caplog.set_level("INFO", logger=stage2_analysis.__name__)
    output = tmp_path / "extraction"
    old_batch = output / "batches" / "batch_00002"
    old_batch.mkdir(parents=True)
    (old_batch / "row_ids.json").write_text("[0]\n", encoding="utf-8")
    (old_batch / "result.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "row_id": 0,
                        "values": {"performance_status": "2"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (old_batch / "complete.json").write_text(
        json.dumps({"status": "complete", "rows": 1}),
        encoding="utf-8",
    )
    definition = {
        "name": "performance_status",
        "value_type": "ordinal",
        "categories_or_unit": ["0-4"],
    }

    def unexpected_request(_messages, _validate, *, request_kind="interpretation"):
        raise AssertionError("shifted legacy singleton should be relocated")

    frame = extract_rows(
        dataset=pd.DataFrame({"clinical_text": ["Pretreatment ECOG was 2."]}),
        row_ids=[0],
        text_column="clinical_text",
        definitions=[definition],
        output_dir=output,
        request_json=unexpected_request,
        workers=1,
        max_prompt_chars=10_000,
    )

    new_batch = output / "batches" / "batch_00001"
    assert frame.loc[0, "performance_status"] == "2"
    assert json.loads((new_batch / "row_ids.json").read_text(encoding="utf-8")) == [0]
    completion = json.loads((new_batch / "complete.json").read_text(encoding="utf-8"))
    assert completion["relocated_single_patient_checkpoint"] is True
    assert completion["checkpoint_source"] == str(old_batch)
    assert "relocate single-patient" in caplog.text


def test_resume_reconstructs_failure_ledger_from_legacy_category_audit(
    tmp_path: Path,
):
    output = tmp_path / "extraction"
    batch = output / "batches" / "batch_00001"
    batch.mkdir(parents=True)
    (batch / "row_ids.json").write_text("[0]\n", encoding="utf-8")
    (batch / "result.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "row_id": 0,
                        "values": {"marker_status": None},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (batch / "complete.json").write_text(
        json.dumps({"status": "complete", "rows": 1}),
        encoding="utf-8",
    )
    (batch / "category_ontology_repair.json").write_text(
        json.dumps(
            {
                "schema_version": "stage2_category_ontology_repair_v1",
                "resolution": "conservative_null",
                "items": [
                    {
                        "mapping_id": "category_mapping_0001",
                        "feature_name": "marker_status",
                        "value_type": "categorical",
                        "allowed_categories": ["negative", "positive"],
                        "prior_extracted_value": "equivocal",
                    }
                ],
                "targets": {
                    "category_mapping_0001": [{"row_id": 0, "feature_name": "marker_status"}]
                },
                "corrections": [{"mapping_id": "category_mapping_0001", "value": None}],
            }
        ),
        encoding="utf-8",
    )
    definition = {
        "name": "marker_status",
        "value_type": "categorical",
        "categories_or_unit": ["negative", "positive"],
    }

    frame = extract_rows(
        dataset=pd.DataFrame({"clinical_text": ["Marker was equivocal."]}),
        row_ids=[0],
        text_column="clinical_text",
        definitions=[definition],
        output_dir=output,
        request_json=(
            lambda _messages, _validate, *, request_kind="interpretation": (_ for _ in ()).throw(
                AssertionError("legacy checkpoint should be adopted")
            )
        ),
        workers=1,
        max_prompt_chars=10_000,
    )

    assert pd.isna(frame.loc[0, "marker_status"])
    ledger = json.loads((batch / "extraction_issues.json").read_text(encoding="utf-8"))
    assert ledger["reconstructed_from_legacy_audits"] is True
    assert ledger["events"][0]["failure_kind"] == "out_of_ontology_category"
    summary = json.loads((output / "failure_summary.json").read_text(encoding="utf-8"))
    assert summary["feature_failure_patterns"][0]["patient_count"] == 1


def test_resume_retries_only_checkpoints_with_stale_range_ontology_repairs(tmp_path: Path):
    output = tmp_path / "extraction"
    batch = output / "batches" / "batch_00001"
    batch.mkdir(parents=True)
    (batch / "result.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "row_id": 0,
                        "values": {"performance_status": None},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (batch / "complete.json").write_text(
        json.dumps({"status": "complete", "rows": 1}),
        encoding="utf-8",
    )
    (batch / "category_ontology_repair.json").write_text(
        json.dumps(
            {
                "schema_version": "stage2_category_ontology_repair_v1",
                "resolution": "conservative_null",
                "items": [
                    {
                        "mapping_id": "category_mapping_0001",
                        "feature_name": "performance_status",
                        "value_type": "ordinal",
                        "allowed_categories": ["0-4"],
                        "prior_extracted_value": 2,
                    }
                ],
                "corrections": [{"mapping_id": "category_mapping_0001", "value": None}],
            }
        ),
        encoding="utf-8",
    )
    definition = {
        "name": "performance_status",
        "value_type": "ordinal",
        "categories_or_unit": ["0-4"],
    }
    calls = []

    def request_json(messages, validate, *, request_kind="interpretation"):
        calls.append(prompt_inputs(messages)["job"])
        return validate(
            {
                "rows": [
                    {
                        "row_id": 0,
                        "values": {"performance_status": 2},
                    }
                ]
            }
        )

    frame = extract_rows(
        dataset=pd.DataFrame({"clinical_text": ["Pretreatment ECOG was 2."]}),
        row_ids=[0],
        text_column="clinical_text",
        definitions=[definition],
        output_dir=output,
        request_json=request_json,
        workers=1,
        max_prompt_chars=10_000,
    )

    assert calls == ["extract_stage2_patient_variables"]
    assert frame.loc[0, "performance_status"] == "2"
    audit = json.loads((batch / "category_ontology_repair.json").read_text(encoding="utf-8"))
    assert audit["resolution"] == "superseded_by_expanded_category_ontology"
    assert audit["previous_audit"]["resolution"] == "conservative_null"


def test_extraction_uses_note_free_category_ontology_after_fifteen_failed_repairs(
    tmp_path: Path,
):
    note = "PRIVATE_NOTE_SENTINEL: prior immunotherapy was documented."
    dataset = pd.DataFrame({"clinical_text": [note]})
    definition = {
        "feature_id": "outer_001_feature_001",
        "name": "prior_immunotherapy_history",
        "description": "Whether prior immunotherapy was documented.",
        "value_type": "binary",
        "categories_or_unit": ["not documented", "documented"],
        "measurement_definition": "Extract documented pretreatment immunotherapy history.",
        "missing_value_rule": "Return null when the history is unavailable.",
    }
    jobs = []
    ontology_body = None

    def completion(messages, _config):
        nonlocal ontology_body
        body = prompt_inputs(messages)
        jobs.append(body["job"])
        if body["job"] == "extract_stage2_patient_variables":
            return json.dumps(
                {
                    "rows": [
                        {
                            "row_id": 0,
                            "values": {"prior_immunotherapy_history": 1},
                        }
                    ]
                }
            )
        assert body["job"] == "map_extracted_values_to_declared_category_ontology"
        ontology_body = body
        assert note not in json.dumps(messages)
        item = body["items"][0]
        return json.dumps(
            {
                "corrections": [
                    {
                        "mapping_id": item["mapping_id"],
                        "value": "documented",
                    }
                ]
            }
        )

    config = PlainHandoffStage2Config(
        endpoint="http://stage2.test/v1",
        model="test-model",
    )
    frame = extract_rows(
        dataset=dataset,
        row_ids=[0],
        text_column="clinical_text",
        definitions=[definition],
        output_dir=tmp_path / "extraction",
        request_json=lambda messages, validate, *, request_kind="interpretation": (
            stage2_workflow._request_json(
                messages=messages,
                config=config,
                completion=completion,
                validate=validate,
                request_kind=request_kind,
            )
        ),
        workers=1,
        max_prompt_chars=config.max_prompt_chars,
    )

    assert jobs == ["extract_stage2_patient_variables"] * 16 + [
        "map_extracted_values_to_declared_category_ontology"
    ]
    assert ontology_body is not None
    assert ontology_body["items"][0]["prior_extracted_value"] == 1
    assert ontology_body["items"][0]["allowed_categories"] == [
        "not documented",
        "documented",
    ]
    assert frame.loc[0, "prior_immunotherapy_history"] == "documented"
    audit = json.loads(
        (
            tmp_path / "extraction" / "batches" / "batch_00001" / "category_ontology_repair.json"
        ).read_text(encoding="utf-8")
    )
    assert audit["resolution"] == "llm_category_ontology"
    assert audit["corrections"][0]["value"] == "documented"


def test_pending_category_ontology_resumes_without_repeating_extraction(
    tmp_path: Path,
):
    dataset = pd.DataFrame({"clinical_text": ["Prior immunotherapy was documented."]})
    definition = {
        "feature_id": "outer_001_feature_001",
        "name": "prior_immunotherapy_history",
        "description": "Whether prior immunotherapy was documented.",
        "value_type": "binary",
        "categories_or_unit": ["not documented", "documented"],
        "measurement_definition": "Extract pretreatment immunotherapy history.",
        "missing_value_rule": "Return null when unavailable.",
    }
    output = tmp_path / "extraction"
    first_calls = []

    def extraction_then_switch(messages, validate, *, request_kind="interpretation"):
        first_calls.append(request_kind)
        if request_kind == "extraction":
            return validate(
                {
                    "rows": [
                        {
                            "row_id": 0,
                            "values": {"prior_immunotherapy_history": 1},
                        }
                    ]
                }
            )
        raise RuntimeError("switch to the interpretation model")

    with pytest.raises(RuntimeError, match="switch to the interpretation model"):
        extract_rows(
            dataset=dataset,
            row_ids=[0],
            text_column="clinical_text",
            definitions=[definition],
            output_dir=output,
            request_json=extraction_then_switch,
            workers=1,
            max_prompt_chars=10_000,
        )

    batch = output / "batches" / "batch_00001"
    pending_path = batch / "pending_category_ontology.json"
    assert first_calls == ["extraction", "interpretation"]
    assert pending_path.is_file()
    resumed_calls = []

    def resume_interpretation(messages, validate, *, request_kind="interpretation"):
        resumed_calls.append(request_kind)
        assert request_kind == "interpretation"
        item = prompt_inputs(messages)["items"][0]
        return validate(
            {
                "corrections": [
                    {
                        "mapping_id": item["mapping_id"],
                        "value": "documented",
                    }
                ]
            }
        )

    frame = extract_rows(
        dataset=dataset,
        row_ids=[0],
        text_column="clinical_text",
        definitions=[definition],
        output_dir=output,
        request_json=resume_interpretation,
        workers=1,
        max_prompt_chars=10_000,
    )

    assert resumed_calls == ["interpretation"]
    assert frame.loc[0, "prior_immunotherapy_history"] == "documented"
    assert not pending_path.exists()
    assert (batch / "complete.json").is_file()


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


def test_extraction_defaults_unmappable_category_to_null_instead_of_crashing(
    tmp_path: Path,
):
    dataset = pd.DataFrame({"clinical_text": ["Prior immunotherapy was documented."]})
    definition = {
        "feature_id": "outer_001_feature_001",
        "name": "prior_immunotherapy_history",
        "value_type": "binary",
        "categories_or_unit": ["not documented", "documented"],
    }

    def request_json(messages, validate, *, request_kind="interpretation"):
        body = prompt_inputs(messages)
        if body["job"] == "extract_stage2_patient_variables":
            return validate(
                {
                    "rows": [
                        {
                            "row_id": 0,
                            "values": {"prior_immunotherapy_history": 1},
                        }
                    ]
                }
            )
        raise stage2_analysis.Stage2ResponseValidationError(
            "category ontology response remained invalid"
        )

    output = tmp_path / "extraction"
    frame = extract_rows(
        dataset=dataset,
        row_ids=[0],
        text_column="clinical_text",
        definitions=[definition],
        output_dir=output,
        request_json=request_json,
        workers=1,
        max_prompt_chars=10_000,
    )

    assert pd.isna(frame.loc[0, "prior_immunotherapy_history"])
    audit = json.loads(
        (output / "batches" / "batch_00001" / "category_ontology_repair.json").read_text(
            encoding="utf-8"
        )
    )
    assert audit["resolution"] == "conservative_null"
    assert "category ontology response remained invalid" in audit["ontology_validation_error"]
    assert audit["corrections"][0]["value"] is None


def test_extraction_defaults_structurally_invalid_response_to_audited_null(
    tmp_path: Path,
    caplog,
):
    caplog.set_level("WARNING", logger=stage2_analysis.__name__)
    output = tmp_path / "extraction"
    definition = {
        "name": "performance_status",
        "value_type": "ordinal",
        "categories_or_unit": ["0-4"],
    }

    def request_json(_messages, _validate, *, request_kind="interpretation"):
        raise stage2_analysis.Stage2ResponseValidationError(
            "Stage 2 response remained invalid after 10 repairs: "
            "Unterminated string at character 104409"
        )

    frame = extract_rows(
        dataset=pd.DataFrame({"clinical_text": ["Pretreatment ECOG was 2."]}),
        row_ids=[0],
        text_column="clinical_text",
        definitions=[definition],
        output_dir=output,
        request_json=request_json,
        workers=1,
        max_prompt_chars=10_000,
    )

    assert pd.isna(frame.loc[0, "performance_status"])
    audit = json.loads(
        (output / "batches" / "batch_00001" / "extraction_failure.json").read_text(encoding="utf-8")
    )
    assert audit["resolution"] == "conservative_all_null"
    assert audit["row_ids"] == [0]
    assert audit["feature_names"] == ["performance_status"]
    assert "Unterminated string" in audit["validation_error"]
    assert "remained structurally invalid" in caplog.text
    failure_summary = json.loads((output / "failure_summary.json").read_text(encoding="utf-8"))
    assert failure_summary["feature_failure_patterns"] == []
    assert failure_summary["structural_failure_patient_count"] == 1


def test_extraction_nulls_only_invalid_feature_value_and_retains_valid_values(
    tmp_path: Path,
):
    dataset = pd.DataFrame({"clinical_text": ["Age 67 years. Blood pressure 147/93 mmHg."]})
    definitions = [
        {
            "feature_id": "outer_001_feature_001",
            "name": "age",
            "value_type": "continuous",
            "categories_or_unit": ["years"],
        },
        {
            "feature_id": "outer_001_feature_002",
            "name": "blood_pressure",
            "value_type": "continuous",
            "categories_or_unit": ["mmHg"],
        },
    ]

    def request_json(_messages, validate, *, request_kind="interpretation"):
        return validate(
            {
                "rows": [
                    {
                        "row_id": 0,
                        "values": {
                            "age": 67,
                            "blood_pressure": {"systolic": 147, "diastolic": 93},
                        },
                    }
                ]
            }
        )

    output = tmp_path / "extraction"
    frame = extract_rows(
        dataset=dataset,
        row_ids=[0],
        text_column="clinical_text",
        definitions=definitions,
        output_dir=output,
        request_json=request_json,
        workers=1,
        max_prompt_chars=10_000,
    )

    assert frame.loc[0, "age"] == 67.0
    assert pd.isna(frame.loc[0, "blood_pressure"])
    assert not (output / "batches" / "batch_00001" / "extraction_failure.json").exists()
    audit = json.loads(
        (output / "batches" / "batch_00001" / "invalid_feature_value_repair.json").read_text(
            encoding="utf-8"
        )
    )
    assert audit["resolution"] == "conservative_invalid_features_null"
    assert audit["issues"][0]["feature_name"] == "blood_pressure"
    assert "dict" in audit["issues"][0]["reason"]


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


def test_discovery_attaches_trusted_provenance_and_preserves_name_collisions():
    rows = [
        {"name": "Risk score", "description": "First distinct scale.", "basis": "First scale is reported.", "uncertainty": ""},
        {"name": "Risk-score", "description": "Second distinct scale.", "basis": "Second scale is reported.", "uncertainty": "Scale is not otherwise named."},
    ]
    result = stage2_workflow._validate_discovery(
        {"candidates": rows}, packet_id="trusted-card", evidence_axes=["outcome", "outcome"],
    )
    assert len({row["name"] for row in result["concepts"]}) == 2
    assert {row["description"] for row in result["concepts"]} == {row["description"] for row in rows}
    assert all(row["supporting_packet_ids"] == ["trusted-card"] for row in result["concepts"])
    assert all(row["evidence_axes"] == ["outcome"] for row in result["concepts"])
    assert result == stage2_workflow._validate_discovery(
        {"candidates": list(reversed(rows))}, packet_id="trusted-card", evidence_axes=["outcome"],
    )
    empty = stage2_workflow._validate_discovery({"candidates": []}, packet_id="empty-card")
    assert empty["packet_dispositions"]["empty-card"]["status"] == "reviewed_no_specific_concept"


@pytest.mark.parametrize("bad_row", [
    {"name": "Clinical finding", "description": "Finding.", "basis": "Explicit.", "uncertainty": "", "supporting_packet_ids": ["forged"]},
    {"name": "Clinical finding", "description": "Finding.", "basis": "", "uncertainty": ""},
    {"name": "Clinical finding", "description": "Finding.", "basis": "Explicit.", "uncertainty": None},
])
def test_discovery_rejects_invalid_content_instead_of_silently_losing_features(bad_row):
    with pytest.raises(ValueError):
        stage2_workflow._validate_discovery({"candidates": [bad_row]}, packet_id="trusted")


def test_rejected_packet_audit_prompt_is_generic_recall_guardrail():
    from oci.inference.stage2_prompt_catalog import SYSTEM_PROMPTS
    excerpt = "pretreatment serum albumin 2.8 g/dL"
    messages = stage2_workflow._rejected_packet_audit_prompt(architecture="bow_r_loss", packets=[{
        "packet_id": "PRIVATE_PACKET", "content": {"representative_evidence": [{"text": excerpt}]}}])
    assert messages[0]["content"] == SYSTEM_PROMPTS["02_audit_unmapped"]
    assert excerpt in messages[1]["content"]
    assert "PRIVATE_PACKET" not in str(messages)
    assert "supporting_items" not in str(messages)
    result = stage2_workflow._validate_interpretation({"candidates": [{"name": "Serum albumin", "description": "Albumin concentration.", "basis": "Albumin result.", "uncertainty": ""}]}, packet_ids=["PRIVATE_PACKET"])
    assert result["concepts"][0]["supporting_packet_ids"] == ["PRIVATE_PACKET"]


def test_operationalization_prompt_prefers_realistic_continuous_measurements():
    from oci.inference.stage2_prompt_catalog import SYSTEM_PROMPTS
    excerpt = "PD-L1 tumor proportion score was 80 percent."
    messages = stage2_workflow._operationalization_prompt(feature_name="pd_l1_expression_level", supporting_evidence=[excerpt])
    assert messages[0]["content"] == SYSTEM_PROMPTS["04_define_ontology"]
    assert "pd l1 expression level" in messages[1]["content"] and excerpt in messages[1]["content"]
    for field in ("feature_id", "outer_fold", "clinical_question", "packet_id", "supporting_items"):
        assert field not in str(messages)


def test_supporting_evidence_preserves_text_and_architecture_without_packet_metadata():
    packets = [
        {
            "packet_id": "opaque_packet_id",
            "architecture": "opaque_architecture",
            "content": {
                "evidence_kind": "clinical_text",
                "semantic_grouping": "opaque_grouping",
                "score_summary": {"score": {"maximum": 9.0}},
                "source_architectures": ["opaque_architecture"],
                "representative_evidence": [
                    {
                        "text": "Readable clinical evidence.",
                        "text_truncated": True,
                        "details": {"opaque": "metadata"},
                    },
                    {"text": "Readable clinical evidence."},
                    {"text": "Second clinical clue."},
                ],
            },
        }
    ]

    assert stage2_workflow._readable_supporting_text(packets) == [
        "Readable clinical evidence.",
        "Second clinical clue.",
    ]
    assert stage2_workflow._packet_support_architectures(packets[0]) == [
        "opaque_architecture"
    ]
    del packets[0]["content"]["source_architectures"]
    assert stage2_workflow._packet_support_architectures(packets[0]) == [
        "opaque_architecture"
    ]


def test_interpretation_normalizes_axes_and_derives_complete_dispositions():
    packet_ids = ["packet-a", "packet-b"]
    result = stage2_workflow._validate_interpretation(
        {
            "candidates": [
                {
                    "name": "performance_status",
                    "supporting_items": [1, 99],
                    "evidence_rationale": (
                        "Functional limitation could produce the cited text pattern."
                    ),
                }
            ],
        },
        packet_ids=packet_ids,
        packet_evidence_axes={
            "packet-a": ["effect-modifier", "unsupported-axis"],
            "packet-b": ["outcome"],
        },
    )

    assert result["concepts"] == [
        {
            "name": "performance_status",
            "description": "performance_status",
            "supporting_packet_ids": ["packet-a"],
            "evidence_axes": ["residual_effect"],
            "evidence_rationale": "Functional limitation could produce the cited text pattern.",
            "caveats": "",
        }
    ]
    assert set(result["packet_dispositions"]) == set(packet_ids)
    assert result["packet_dispositions"]["packet-a"]["status"] == "supports_concept"
    assert result["packet_dispositions"]["packet-b"]["status"] == "reviewed_no_specific_concept"


def test_interpretation_requires_a_latent_feature_evidence_rationale():
    with pytest.raises(ValueError, match="has no evidence_rationale"):
        stage2_workflow._validate_interpretation(
            {
                "concepts": [
                    {
                        "name": "performance_status",
                        "supporting_items": [1],
                    }
                ],
            },
            packet_ids=["packet-a"],
        )


def test_interpretation_ignores_unnamed_candidate_and_preserves_valid_candidates(caplog):
    result = stage2_workflow._validate_interpretation(
        {
            "candidates": [
                {
                    "name": "",
                    "description": "An ambiguous fragment with no defensible feature name.",
                    "supporting_items": [1],
                    "evidence_rationale": "The fragment was reviewed but is not identifiable.",
                },
                {
                    "name": "thrombocytopenia",
                    "supporting_items": [2],
                    "evidence_rationale": "The cited text explicitly names thrombocytopenia.",
                },
            ]
        },
        packet_ids=["packet-ambiguous", "packet-thrombocytopenia"],
    )

    assert [concept["name"] for concept in result["concepts"]] == ["thrombocytopenia"]
    assert (
        result["packet_dispositions"]["packet-ambiguous"]["status"]
        == "reviewed_no_specific_concept"
    )
    assert result["packet_dispositions"]["packet-thrombocytopenia"]["status"] == "supports_concept"
    assert "ignored unnamed candidate at position=1" in caplog.text


def test_interpretation_drops_only_concepts_without_grounded_packet_citations(caplog):
    result = stage2_workflow._validate_interpretation(
        {
            "concepts": [
                {
                    "name": "performance_status",
                    "supporting_items": [1],
                    "evidence_rationale": "Functional limitation could explain the evidence.",
                },
                {
                    "name": "invented_feature",
                    "supporting_items": [99],
                },
            ],
        },
        packet_ids=["packet-a", "packet-b"],
    )

    assert [concept["name"] for concept in result["concepts"]] == ["performance_status"]
    assert set(result["packet_dispositions"]) == {"packet-a", "packet-b"}
    assert "dropped ungrounded concept=invented_feature" in caplog.text


def test_interpretation_maps_prompt_local_numeric_string_to_packet_id():
    result = stage2_workflow._validate_interpretation(
        {
            "concepts": [
                {
                    "name": "performance_status",
                    "supporting_items": ["1"],
                    "evidence_rationale": "Functional limitation could explain the evidence.",
                }
            ],
        },
        packet_ids=["packet-a"],
    )

    assert result["concepts"][0]["supporting_packet_ids"] == ["packet-a"]


def test_interpretation_second_pass_recovers_rejected_named_measurement(tmp_path: Path):
    packet = {
        "packet_id": "outer_001_card_albumin",
        "architecture": "bow_r_loss",
        "outer_fold": 1,
        "observable_axes": ["residual_effect"],
        "content": {
            "evidence_kind": "lexical_term",
            "representative_evidence": [{"text": "pretreatment serum albumin 2.8 g/dL"}],
            "support": {"inner_folds": [1, 2, 3, 4, 5]},
        },
    }
    calls = []
    secret_question = "SECRET CLINICAL QUESTION THAT MUST NOT REACH INTERPRETATION"

    def completion(messages, _config):
        assert secret_question not in messages[1]["content"]
        if messages[0]["content"].startswith("You identify clinical variables") and not calls:
            calls.append("infer_clinical_features_from_text_evidence")
            return json.dumps({"candidates": []})
        body = prompt_inputs(messages)
        calls.append(body["job"])
        assert "clinical_question" not in body
        assert body["job"] == "audit_unmapped_text_evidence_for_missed_clinical_features"
        return json.dumps(
            {
                "candidates": [
                    {
                        "name": "serum_albumin",
                        "description": "Serum albumin concentration.",
                        "supporting_items": [1],
                        "evidence_rationale": (
                            "The evidence item directly names a reproducibly measurable "
                            "laboratory value."
                        ),
                        "caveats": "Confirm units during extraction.",
                    }
                ],
            }
        )

    runner = PlainHandoffStage2(
        config=PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
        ),
        clinical_question=secret_question,
        completion=completion,
    )
    output_dir = tmp_path / "batch_001"

    result = runner._interpret_batch(
        architecture=packet["architecture"],
        packets=[packet],
        output_dir=output_dir,
    )

    assert calls == [
        "infer_clinical_features_from_text_evidence",
        "audit_unmapped_text_evidence_for_missed_clinical_features",
    ]
    assert [concept["name"] for concept in result["concepts"]] == ["serum_albumin"]
    assert result["packet_dispositions"][packet["packet_id"]]["status"] == ("supports_concept")
    assert result["rejected_packet_audit"]["recovered_packet_ids"] == [packet["packet_id"]]
    assert not result["rejected_packet_audit"]["remaining_rejected_packet_ids"]
    assert (output_dir / "rejected_packet_audit" / "batch_001" / "complete.json").is_file()
    request_paths = [
        output_dir / "input.json",
        output_dir / "initial" / "input.json",
        output_dir / "rejected_packet_audit" / "batch_001" / "input.json",
    ]
    for request_path in request_paths:
        saved_input = json.loads(request_path.read_text(encoding="utf-8"))
        assert "clinical_question" not in saved_input
        assert secret_question not in json.dumps(saved_input)

    cached = runner._interpret_batch(
        architecture=packet["architecture"],
        packets=[packet],
        output_dir=output_dir,
    )

    assert calls == [
        "infer_clinical_features_from_text_evidence",
        "audit_unmapped_text_evidence_for_missed_clinical_features",
    ]
    assert cached == result

    def should_not_call(_messages, _config):
        raise AssertionError("endpoint-only change should reuse the checkpoint")

    moved_runner = PlainHandoffStage2(
        config=PlainHandoffStage2Config(
            endpoint="http://replacement-stage2.test/v1",
            model="test-model",
        ),
        clinical_question=secret_question,
        completion=should_not_call,
    )
    assert moved_runner._interpret_batch(
        architecture=packet["architecture"],
        packets=[packet],
        output_dir=output_dir,
    ) == result


def test_interpretation_does_not_pair_new_input_with_an_old_complete_result(
    tmp_path: Path,
):
    packet = {
        "packet_id": "outer_001_card_current",
        "architecture": "semantic_clustered_clinical_text",
        "outer_fold": 1,
        "observable_axes": ["outcome"],
        "content": {"representative_evidence": [{"text": "ECOG 2"}]},
    }
    calls = []

    def completion(_messages, _config):
        calls.append("called")
        return json.dumps({"candidates": [{
            "name": "Performance status",
            "description": "ECOG performance status.",
            "basis": "The text explicitly reports ECOG 2.",
            "uncertainty": "",
        }]})

    runner = PlainHandoffStage2(
        config=PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
        ),
        clinical_question="Identify confounders.",
        completion=completion,
    )
    output_dir = tmp_path / "batch_001"
    output_dir.mkdir()
    input_value = {
        "interpretation_schema": stage2_workflow.INTERPRETATION_SCHEMA_VERSION,
        "llm_identity": {
            "model": "test-model",
        },
        "architecture": packet["architecture"],
        "packets": [packet],
    }
    (output_dir / "input.json").write_text(
        json.dumps(
            {
                **input_value,
                "input_fingerprint": stage2_workflow._value_fingerprint(input_value),
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "result.json").write_text(
        json.dumps(
            {
                "concepts": [
                    {
                        "name": "stale_concept",
                        "supporting_packet_ids": ["outer_001_row_0002_section_02_part_001"],
                    }
                ],
                "packet_dispositions": {
                    "outer_001_row_0002_section_02_part_001": {"status": "supports_concept"}
                },
            }
        ),
        encoding="utf-8",
    )
    # This marker belongs to the old result. A prior interrupted rerun had
    # already replaced input.json but had not produced a new result/marker.
    (output_dir / "complete.json").write_text(
        json.dumps({"status": "complete"}),
        encoding="utf-8",
    )

    result = runner._interpret_batch(
        architecture=packet["architecture"],
        packets=[packet],
        output_dir=output_dir,
    )

    assert calls == ["called"]
    assert result["concepts"][0]["supporting_packet_ids"] == [packet["packet_id"]]
    completion_state = json.loads((output_dir / "complete.json").read_text(encoding="utf-8"))
    assert completion_state["input_fingerprint"] == stage2_workflow._value_fingerprint(input_value)


def test_stage2_retries_retryable_transport_errors_without_using_repair_turns(
    monkeypatch,
):
    class RetryableTransportError(Exception):
        pass

    calls = []
    delays = []

    def completion(messages, _config):
        calls.append([dict(message) for message in messages])
        if len(calls) < 3:
            raise RetryableTransportError("temporary timeout")
        return '{"ok": true}'

    monkeypatch.setattr(
        stage2_workflow,
        "_is_retryable_transport_error",
        lambda exc: isinstance(exc, RetryableTransportError),
    )
    monkeypatch.setattr(stage2_workflow.time, "sleep", delays.append)
    config = PlainHandoffStage2Config(
        endpoint="http://stage2.test/v1",
        model="test-model",
        transport_max_attempts=3,
        transport_retry_backoff=0.25,
    )

    result = stage2_workflow._request_json(
        messages=[{"role": "user", "content": "Return JSON."}],
        config=config,
        completion=completion,
        validate=lambda value: dict(value),
    )

    assert result == {"ok": True}
    assert calls[1] == calls[2]
    assert calls[1][:-1] == calls[0]
    assert "RetryableTransportError: temporary timeout" in calls[1][-1]["content"]
    assert delays == [0.25, 0.5]


def test_stage2_does_not_reclassify_a_pre_response_failure_as_invalid_science():
    calls = []

    def completion(_messages, _config):
        calls.append("called")
        raise ValueError("endpoint routing is misconfigured")

    with pytest.raises(ValueError, match="endpoint routing is misconfigured"):
        stage2_workflow._request_json(
            messages=[{"role": "user", "content": "Return JSON."}],
            config=PlainHandoffStage2Config(
                endpoint="http://stage2.test/v1",
                model="test-model",
            ),
            completion=completion,
            validate=lambda value: dict(value),
        )

    assert calls == ["called"]


def test_stage2_logical_request_deadline_bounds_transport_retries(monkeypatch):
    class RetryableTransportError(Exception):
        pass

    clock = [0.0]
    attempt_timeouts = []

    def completion(_messages, request_config):
        attempt_timeouts.append(request_config.request_timeout)
        raise RetryableTransportError("straggling endpoint")

    monkeypatch.setattr(
        stage2_workflow,
        "_is_retryable_transport_error",
        lambda exc: isinstance(exc, RetryableTransportError),
    )
    monkeypatch.setattr(stage2_workflow.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        stage2_workflow.time,
        "sleep",
        lambda delay: clock.__setitem__(0, clock[0] + delay),
    )
    config = PlainHandoffStage2Config(
        endpoint="http://stage2.test/v1",
        model="test-model",
        request_timeout=2.5,
        request_attempt_timeout=1.0,
        transport_max_attempts=10,
        transport_retry_backoff=1.0,
    )

    with pytest.raises(
        stage2_analysis.Stage2RequestExhaustedError,
        match="deadline",
    ):
        stage2_workflow._request_json(
            messages=[{"role": "user", "content": "Return JSON."}],
            config=config,
            completion=completion,
            validate=lambda value: dict(value),
        )

    assert attempt_timeouts == [1.0, 1.0]
    assert clock[0] == 1.0


@pytest.mark.parametrize(
    ("request_kind", "reasoning_effort", "max_tokens"),
        [
            ("interpretation", "high", 100_000),
            ("extraction", "none", 75_000),
    ],
)
def test_openai_completion_sends_request_scoped_reasoning_and_token_cap(
    monkeypatch,
    request_kind,
    reasoning_effort,
    max_tokens,
):
    request_kwargs = {}

    class FakeCompletions:
        @staticmethod
        def create(**kwargs):
            request_kwargs.update(kwargs)
            message = type("Message", (), {"content": '{"ok": true}'})()
            choice = type("Choice", (), {"message": message, "finish_reason": "stop"})()
            return type("Response", (), {"choices": [choice]})()

    class FakeClient:
        def __init__(self):
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()
            self.closed = False

        def close(self):
            self.closed = True

    client = FakeClient()
    client_kwargs = {}
    import openai

    def fake_client(**kwargs):
        client_kwargs.update(kwargs)
        return client

    monkeypatch.setattr(openai, "OpenAI", fake_client)
    content = stage2_workflow._openai_completion(
        [{"role": "user", "content": "Return JSON."}],
        PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
            runtime_request_kind=request_kind,
        ),
    )

    assert content == '{"ok": true}'
    assert client.closed is True
    assert client_kwargs["max_retries"] == 0
    assert request_kwargs["reasoning_effort"] == reasoning_effort
    assert "extra_body" not in request_kwargs
    assert request_kwargs["max_tokens"] == max_tokens
    assert "max_completion_tokens" not in request_kwargs


@pytest.mark.parametrize(
    (
        "request_kind",
        "enabled",
        "soft_switch",
        "wire_reasoning_effort",
        "expected_max_tokens",
    ),
        [
            ("extraction", False, "/no_think", None, 75_000),
        ("interpretation", True, "/think", "xhigh", 100_000),
    ],
)
def test_qwen38_requests_use_hard_and_soft_thinking_controls(
    monkeypatch,
    request_kind,
    enabled,
    soft_switch,
    wire_reasoning_effort,
    expected_max_tokens,
):
    request_kwargs = {}

    class FakeCompletions:
        @staticmethod
        def create(**kwargs):
            request_kwargs.update(kwargs)
            message = type("Message", (), {"content": '{"ok": true}'})()
            choice = type("Choice", (), {"message": message, "finish_reason": "stop"})()
            return type("Response", (), {"choices": [choice]})()

    class FakeClient:
        def __init__(self, **_kwargs):
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

        def close(self):
            pass

    import openai

    monkeypatch.setattr(openai, "OpenAI", FakeClient)
    result = stage2_workflow._openai_completion(
        [{"role": "user", "content": "Return JSON."}],
        PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="Qwen/Qwen3.8-27B",
            runtime_request_kind=request_kind,
        ),
    )

    assert result == '{"ok": true}'
    assert request_kwargs["max_tokens"] == expected_max_tokens
    assert request_kwargs["messages"][-1]["content"].endswith(soft_switch)
    assert request_kwargs["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": enabled
    }
    if wire_reasoning_effort is None:
        assert "reasoning_effort" not in request_kwargs
    else:
        assert request_kwargs["reasoning_effort"] == wire_reasoning_effort


@pytest.mark.parametrize(
    ("configured_effort", "wire_effort"),
    [
        ("none", None),
        ("minimal", None),
        ("low", "low"),
        ("medium", "medium"),
        ("high", "xhigh"),
        ("xhigh", "xhigh"),
        ("max", "xhigh"),
    ],
)
def test_qwen38_reasoning_effort_uses_supported_wire_vocabulary(
    configured_effort,
    wire_effort,
):
    assert stage2_workflow._wire_reasoning_effort(
        configured_effort=configured_effort,
        model_family="qwen3",
    ) == wire_effort


def test_openai_completion_falls_back_when_reasoning_effort_is_not_supported(
    monkeypatch,
):
    calls = []

    class UnsupportedParameterError(Exception):
        status_code = 400

    class FakeCompletions:
        @staticmethod
        def create(**kwargs):
            calls.append(kwargs)
            if "reasoning_effort" in kwargs:
                raise UnsupportedParameterError(
                    "Unexpected reasoning effort high. Supported types are xhigh, medium, and low."
                )
            message = type("Message", (), {"content": '{"ok": true}'})()
            choice = type("Choice", (), {"message": message, "finish_reason": "stop"})()
            return type("Response", (), {"choices": [choice]})()

    class FakeClient:
        def __init__(self, **_kwargs):
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

        def close(self):
            pass

    import openai

    monkeypatch.setattr(openai, "OpenAI", FakeClient)

    assert stage2_workflow._openai_completion(
        [{"role": "user", "content": "Return JSON."}],
        PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="google/gemma-4-26B-A4B-it",
        ),
    ) == '{"ok": true}'
    assert len(calls) == 2
    assert "reasoning_effort" in calls[0]
    assert "reasoning_effort" not in calls[1]
    assert calls[1]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True


@pytest.mark.parametrize(
    "response",
    [
        '<think>draft {"draft": true}</think>\n{"ok": true}',
        '<think>```json\n{"draft": true}\n```</think>\n{"ok": true}',
        '<|channel>thought\nconsider alternatives<channel|>{"ok": true}',
        'Reasoning that was not parsed by the server.\n```json\n{"ok": true}\n```',
    ],
)
def test_json_parser_extracts_final_object_after_inline_reasoning(response):
    assert stage2_workflow._parse_json_object(response) == {"ok": True}


def test_response_message_prefers_final_content_over_parsed_reasoning():
    message = type(
        "Message",
        (),
        {
            "content": '{"ok": true}',
            "reasoning_content": 'draft {"ok": false}',
        },
    )()

    assert stage2_workflow._response_message_text(message) == '{"ok": true}'


def test_openai_completion_reports_output_token_truncation(monkeypatch):
    class FakeCompletions:
        @staticmethod
        def create(**_kwargs):
            message = type("Message", (), {"content": '{"rows": ['})()
            choice = type(
                "Choice",
                (),
                {"message": message, "finish_reason": "length"},
            )()
            return type("Response", (), {"choices": [choice]})()

    class FakeClient:
        def __init__(self, **_kwargs):
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()
            self.closed = False

        def close(self):
            self.closed = True

    import openai

    client = FakeClient()
    monkeypatch.setattr(openai, "OpenAI", lambda **_kwargs: client)

    with pytest.raises(ValueError, match="finish_reason=length"):
        stage2_workflow._openai_completion(
            [{"role": "user", "content": "Return JSON."}],
            PlainHandoffStage2Config(
                endpoint="http://stage2.test/v1",
                model="test-model",
            ),
        )

    assert client.closed is True


def _agentic_fixture_response(body):
    if body.get("task") == "adjudicate_stage2_roles_from_all_evidence":
        decisions = []
        for item in body["role_evidence"]["candidates"]:
            definition = item["definition"]
            roles = (
                list(definition["configured_roles"])
                if definition["investigator_locked"]
                else ["confounder", "effect_modifier"]
            )
            decisions.append(
                {
                    "feature_id": item["feature_id"],
                    "roles": roles,
                    "evidence_for": ["Fold-honest fixture evidence."],
                    "evidence_against": [],
                    "inner_fold_consistency": "Consistent in fixture folds.",
                    "cross_method_reconciliation": (
                        "Fixture methods support the assigned roles."
                    ),
                    "rationale": "Exercise all-evidence role adjudication.",
                }
            )
        return {
            "summary": "Apply fixture all-evidence decisions.",
            "decisions": decisions,
        }
    if body.get("task") == "analyze_cluster":
        recommendations = []
        for feature in body["definitions"]:
            promote = (
                body["role"] in feature.get("roles", [])
                if feature.get("configured_explicit_feature") is True
                else True
            )
            recommendations.append(
                {
                    "candidate_id": feature["feature_id"],
                    "promote": promote,
                    "evidence_for": ["test fixture retention"] if promote else [],
                    "evidence_against": [] if promote else ["role not configured"],
                    "inner_fold_consistency": "Consistent in the fixture folds.",
                    "rationale": "Exercise the agentic selection integration.",
                }
            )
        return {
            "action": "final",
            "role": body["role"],
            "cluster_id": body["cluster"]["cluster_id"],
            "assessment": "Apply fixture decisions.",
            "latent_ids": [],
            "recommendations": recommendations,
        }
    if body.get("task") == "outer_fold_role_adjudication":
        decisions = []
        for item in body["eligible_candidates"]:
            definition = item["definition"]
            promote = (
                body["role"] in definition.get("roles", [])
                if definition.get("configured_explicit_feature") is True
                else True
            )
            decisions.append(
                {
                    "candidate_id": item["candidate_id"],
                    "promote": promote,
                    "evidence_for": ["test fixture retention"] if promote else [],
                    "evidence_against": [] if promote else ["role not configured"],
                    "inner_fold_consistency": "Consistent in the fixture folds.",
                    "rationale": "Exercise the agentic adjudication integration.",
                }
            )
        return {
            "action": "final",
            "role": body["role"],
            "summary": "Apply fixture decisions.",
            "decisions": decisions,
            "selected_candidate_ids": [
                row["candidate_id"] for row in decisions if row["promote"]
            ],
            "latent_source_exceptions": [],
        }
    raise AssertionError("not an agentic fixture prompt")


def _fake_completion(calls):
    def complete(messages, _config):
        if messages[0]["content"].startswith("You identify clinical variables"):
            body = {"job": "infer_clinical_features_from_text_evidence"}
        else:
            body = prompt_inputs(messages)
        job = _prompt_job(body)
        calls.append(job)
        if body.get("task") in {
            "analyze_cluster",
            "outer_fold_role_adjudication",
            "adjudicate_stage2_roles_from_all_evidence",
        }:
            return json.dumps(_agentic_fixture_response(body))
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
        if job == "review_stage2_variables_against_training_fold_performance":
            return json.dumps(
                {
                    "feature_decisions": [
                        {
                            "feature_id": feature["feature_id"],
                            "action": "keep",
                            "reason": "The training-fold extraction is usable.",
                        }
                        for feature in body["features"]
                    ],
                    "overall_assessment": "Keep the operational definitions.",
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


def test_packetizer_removes_old_control_plane_fields():
    packets = packetize_handoff(
        [
            {
                "source": "tfidf",
                "outer_fold": 1,
                "scope": "full_outer_train",
                "evidence": {
                    "architecture": "tfidf_topic_contrast",
                    "evidence_id": "topic-1",
                    "terms": ["ECOG", "performance status"],
                    "fit_row_ids": [1, 2, 3],
                    "artifact_inventory": {"path": "/sealed/place"},
                    "content_sha256": "a" * 64,
                },
            }
        ],
        max_packet_chars=2_000,
    )

    assert len(packets) == 1
    assert packets[0]["content"]["terms"] == ["ECOG", "performance status"]
    assert packets[0]["observable_axes"] == ["semantic"]
    assert "fit_row_ids" not in packets[0]["content"]
    assert "artifact_inventory" not in packets[0]["content"]
    assert "content_sha256" not in packets[0]["content"]


def test_packet_axes_use_model_objectives_not_clinical_witness_wording():
    packets = packetize_handoff(
        [
            {
                "source": "text_models",
                "outer_fold": 1,
                "scope": "full_outer_train",
                "evidence": {
                    "architecture": "embedding_contrast_whole",
                    "objective": "outcome",
                    "witnesses": ["ECOG 2 before treatment."],
                },
            }
        ],
        max_packet_chars=2_000,
    )

    assert packets[0]["observable_axes"] == ["outcome", "semantic"]


def test_mixed_tfidf_and_neural_banks_become_separate_axis_packets():
    packets = packetize_handoff(
        [
            {
                "source": "tfidf",
                "outer_fold": 1,
                "scope": "full_outer_train",
                "evidence": {
                    "discovery": {
                        "topic_banks": {
                            "treatment": {"topics": [{"terms": ["cisplatin"]}]},
                            "outcome": {"topics": [{"terms": ["cachexia"]}]},
                        }
                    }
                },
            },
            {
                "source": "neural_queries",
                "outer_fold": 1,
                "scope": "full_outer_train",
                "evidence": {
                    "evidence": [
                        {"bank": "treatment", "witnesses": ["frail"]},
                        {"bank": "effect", "witnesses": ["squamous"]},
                    ]
                },
            },
        ],
        max_packet_chars=2_000,
    )

    axes_by_path = {packet["json_path"]: packet["observable_axes"] for packet in packets}
    assert axes_by_path["discovery.topic_banks.treatment"] == ["semantic", "treatment"]
    assert axes_by_path["discovery.topic_banks.outcome"] == ["outcome", "semantic"]
    assert axes_by_path["evidence.treatment"] == ["semantic", "treatment"]
    assert axes_by_path["evidence.effect"] == ["residual_effect", "semantic"]


def test_packetizer_losslessly_splits_json_expanding_unicode():
    text = "漢" * 5_000
    packets = packetize_handoff(
        [
            {
                "source": "custom",
                "outer_fold": 1,
                "scope": "full_outer_train",
                "evidence": {
                    "architecture": "unicode_witnesses",
                    "payload": text,
                },
            }
        ],
        max_packet_chars=2_000,
    )

    payload_packets = [packet for packet in packets if "payload" in packet["json_path"]]
    assert "".join(str(packet["content"]) for packet in payload_packets) == text
    assert all(
        len(json.dumps(packet, separators=(",", ":"), sort_keys=True)) <= 2_000
        for packet in packets
    )


def test_packetizer_accounts_for_deep_generated_json_paths():
    long_key = "nested_" + ("x" * 500)
    text = "漢" * 5_000
    packets = packetize_handoff(
        [
            {
                "source": "custom",
                "outer_fold": 1,
                "scope": "full_outer_train",
                "evidence": {
                    "architecture": "deep_unicode_witnesses",
                    "nested": {long_key: {"payload": text}},
                },
            }
        ],
        max_packet_chars=2_000,
    )

    payload_packets = [packet for packet in packets if "payload" in packet["json_path"]]
    assert "".join(str(packet["content"]) for packet in payload_packets) == text
    assert all(
        len(json.dumps(packet, separators=(",", ":"), sort_keys=True)) <= 2_000
        for packet in packets
    )


def test_packetizer_splits_large_flat_lists_without_changing_items():
    terms = [f"term_{index:05d}" for index in range(20_000)]
    packets = packetize_handoff(
        [
            {
                "source": "custom",
                "outer_fold": 1,
                "scope": "full_outer_train",
                "evidence": {
                    "architecture": "large_term_inventory",
                    "terms": terms,
                },
            }
        ],
        max_packet_chars=2_000,
    )

    term_packets = [packet for packet in packets if "terms" in packet["json_path"]]
    assert [term for packet in term_packets for term in packet["content"]] == terms
    assert all(
        len(json.dumps(packet, separators=(",", ":"), sort_keys=True)) <= 2_000
        for packet in packets
    )


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


def test_stage2_feature_batch_limit_is_preserved_across_lossless_pages(tmp_path: Path):
    note = "n" * 9_000
    definitions = [
        {
            "name": f"page_feature_{index:02d}",
            "description": "A compact page feature.",
            "value_type": "continuous",
            "categories_or_unit": ["score"],
            "measurement_definition": "Extract the documented score.",
            "missing_value_rule": "Return null when undocumented.",
        }
        for index in range(11)
    ]
    extraction_bodies = []
    prompt_sizes = []

    def request_json(messages, validate, *, request_kind="interpretation"):
        prompt_sizes.append(sum(len(message["content"]) for message in messages))
        body = prompt_inputs(messages)
        names = [feature["name"] for feature in body["features"]]
        assert len(names) <= 4
        assert body["job"] == "extract_stage2_patient_variable_observations"
        extraction_bodies.append(body)
        patient = body["patient"]
        return validate(
            {
                "rows": [
                    {
                        "row_id": patient["row_id"],
                        "observations": [
                            _page_observation(
                                feature_name=name,
                                value=int(name.removeprefix("page_feature_")),
                                text=patient["text"],
                                evidence="n",
                            )
                            for name in names
                        ],
                    }
                ]
            }
        )

    output = tmp_path / "extraction"
    frame = extract_rows(
        dataset=pd.DataFrame({"clinical_text": [note]}),
        row_ids=[0],
        text_column="clinical_text",
        definitions=definitions,
        output_dir=output,
        request_json=request_json,
        workers=3,
        max_prompt_chars=5_000,
        feature_batch_size=4,
    )

    pages_by_index = {}
    for body in extraction_bodies:
        page = body["patient"]
        pages_by_index[int(page["page"]["page_index"])] = page
    ordered_pages = [pages_by_index[index] for index in sorted(pages_by_index)]
    assert len(ordered_pages) >= 2
    assert "".join(page["text"] for page in ordered_pages) == note
    assert all(len(body["features"]) <= 4 for body in extraction_bodies)
    assert all(size <= 5_000 for size in prompt_sizes)
    assert frame.loc[0, "page_feature_10"] == 10.0
    page_completion = json.loads(
        (output / "pages" / "row_00000000" / "page_00001" / "complete.json").read_text(
            encoding="utf-8"
        )
    )
    assert page_completion["feature_batches"] == 3
    reconciliation_completion = json.loads(
        (output / "pages" / "row_00000000" / "reconciliation" / "complete.json").read_text(
            encoding="utf-8"
        )
    )
    assert reconciliation_completion["reconciliation_method"] == "deterministic_provenance"
    assert reconciliation_completion["features"] == 11


def test_stage2_reconciles_oversized_page_observations_without_another_llm_request(
    tmp_path: Path,
):
    note = "n" * 2_500
    definitions = []
    expected_values = {}
    for index in range(2):
        name = f"generic_state_{index}"
        selected = f"state_{index}_" + ("x" * 300)
        expected_values[name] = selected
        definitions.append(
            {
                "feature_id": f"feature_{index}",
                "name": name,
                "description": "d" * 200,
                "value_type": "categorical",
                "categories_or_unit": [selected, f"other_state_{index}"],
                "roles": ["confounder"],
                "measurement_definition": "m" * 350,
                "missing_value_rule": "z" * 100,
            }
        )

    page_bodies = []
    prompt_sizes = []

    def request_json(messages, validate, *, request_kind="interpretation"):
        prompt_sizes.append(sum(len(message["content"]) for message in messages))
        body = prompt_inputs(messages)
        assert body["job"] == "extract_stage2_patient_variable_observations"
        page_bodies.append(body["patient"])
        patient = body["patient"]
        return validate(
            {
                "rows": [
                    {
                        "row_id": patient["row_id"],
                        "observations": [
                            _page_observation(
                                feature_name=feature["name"],
                                value=expected_values[feature["name"]],
                                text=patient["text"],
                                evidence="n",
                            )
                            for feature in body["features"]
                        ],
                    }
                ]
            }
        )

    output = tmp_path / "extraction"
    frame = extract_rows(
        dataset=pd.DataFrame({"clinical_text": [note]}),
        row_ids=[0],
        text_column="clinical_text",
        definitions=definitions,
        output_dir=output,
        request_json=request_json,
        workers=4,
        max_prompt_chars=5_000,
    )

    ordered_pages = sorted(page_bodies, key=lambda row: row["page"]["page_index"])
    assert "".join(row["text"] for row in ordered_pages) == note
    assert len(ordered_pages) >= 2
    assert all(size <= 5_000 for size in prompt_sizes)
    assert frame.loc[0, definitions[0]["name"]] == expected_values[definitions[0]["name"]]
    assert frame.loc[0, definitions[1]["name"]] == expected_values[definitions[1]["name"]]
    completion = json.loads(
        (output / "pages" / "row_00000000" / "reconciliation" / "complete.json").read_text(
            encoding="utf-8"
        )
    )
    assert completion["reconciliation_method"] == "deterministic_provenance"
    assert completion["features"] == 2


def test_stage2_oversized_note_uses_verified_dates_instead_of_page_order(tmp_path: Path):
    newest = "2024-02-01 PD-L1 TPS 50%"
    older = "2023-01-01 PD-L1 TPS 10%"
    note = newest + (" x" * 5_000) + older
    definition = {
        "name": "pd_l1_tps",
        "description": "PD-L1 tumor proportion score.",
        "value_type": "continuous",
        "categories_or_unit": ["percent"],
        "measurement_definition": "Extract the latest documented PD-L1 TPS.",
        "missing_value_rule": "Return null when undocumented.",
        "conflict_resolution": {"strategy": "latest", "positive_category": None},
    }
    page_prompts = []

    def request_json(messages, validate, *, request_kind="interpretation"):
        body = prompt_inputs(messages)
        assert body["job"] == "extract_stage2_patient_variable_observations"
        page_prompts.append(body)
        patient = body["patient"]
        observations = []
        for evidence, value, recorded_at in (
            (newest, 50, "2024-02-01"),
            (older, 10, "2023-01-01"),
        ):
            if evidence in patient["text"]:
                observations.append(
                    _page_observation(
                        feature_name="pd_l1_tps",
                        value=value,
                        text=patient["text"],
                        evidence=evidence,
                        recorded_at=recorded_at,
                        recorded_at_evidence=recorded_at,
                    )
                )
        return validate(
            {
                "rows": [
                    {
                        "row_id": patient["row_id"],
                        "observations": observations,
                    }
                ]
            }
        )

    output = tmp_path / "extraction"
    frame = extract_rows(
        dataset=pd.DataFrame({"clinical_text": [note]}),
        row_ids=[0],
        text_column="clinical_text",
        definitions=[definition],
        output_dir=output,
        request_json=request_json,
        workers=4,
        max_prompt_chars=5_000,
    )

    assert len(page_prompts) >= 2
    assert frame.loc[0, "pd_l1_tps"] == 50.0
    decisions = json.loads(
        (
            output / "pages" / "row_00000000" / "reconciliation" / "decisions.json"
        ).read_text(encoding="utf-8")
    )
    decision = decisions["decisions"]["pd_l1_tps"]
    assert decision["distinct_value_count"] == 2
    assert decision["selection_basis"] == "verified_recorded_at"
    assert decision["value"] == 50.0
    selected = next(
        observation
        for observation in decision["observations"]
        if observation["observation_id"] == decision["selected_observation_id"]
    )
    assert selected["recorded_at"] == "2024-02-01"
    assert note[selected["source_start"] : selected["source_end"]] == newest

    def unexpected_request(*_args, **_kwargs):
        raise AssertionError("completed page observations and reconciliation must resume")

    resumed = extract_rows(
        dataset=pd.DataFrame({"clinical_text": [note]}),
        row_ids=[0],
        text_column="clinical_text",
        definitions=[definition],
        output_dir=output,
        request_json=unexpected_request,
        workers=4,
        max_prompt_chars=5_000,
    )
    assert resumed.loc[0, "pd_l1_tps"] == 50.0


def test_stage2_single_or_null_conflict_policy_is_conservative_and_audited():
    definition = {
        "name": "ambiguous_status",
        "description": "A status with no valid longitudinal precedence.",
        "value_type": "categorical",
        "categories_or_unit": ["A", "B"],
        "measurement_definition": "Return one value only when unambiguous.",
        "missing_value_rule": "Return null for conflicting documentation.",
        "conflict_resolution": {"strategy": "single_or_null"},
    }
    observations = [
        {
            "observation_id": "observation_a",
            "feature_name": "ambiguous_status",
            "value": "A",
            "source_start": 10,
            "source_end": 11,
            "recorded_at": None,
        },
        {
            "observation_id": "observation_b",
            "feature_name": "ambiguous_status",
            "value": "B",
            "source_start": 20,
            "source_end": 21,
            "recorded_at": None,
        },
    ]

    value, decision = stage2_analysis._resolve_feature_observations(
        definition=definition,
        observations=observations,
    )

    assert value is None
    assert decision["resolution"] == "conflict_null"
    assert decision["selection_basis"] == "conflicting_values_are_null"
    assert decision["selected_observation_id"] is None


def test_stage2_page_provenance_requires_exact_quotes_and_repairs_offsets():
    text = "Encounter 2024-06-03: ECOG was 2."
    page = {
        "row_id": 4,
        "text": text,
        "page": {
            "page_index": 2,
            "char_start": 100,
            "char_end": 100 + len(text),
            "document_chars": 300,
        },
    }
    definition = {
        "name": "ecog",
        "description": "ECOG performance status.",
        "value_type": "ordinal",
        "categories_or_unit": ["0", "1", "2", "3", "4"],
        "measurement_definition": "Extract the latest ECOG score.",
        "missing_value_rule": "Return null when undocumented.",
    }
    response = {
        "rows": [
            {
                "row_id": 4,
                "observations": [
                    {
                        "feature_name": "ecog",
                        "value": "2",
                        "evidence": "ECOG was 2",
                        "evidence_start": 0,
                        "evidence_end": 2,
                        "recorded_at": "2024-06-03",
                        "recorded_at_evidence": "2024-06-03",
                        "recorded_at_start": 0,
                        "recorded_at_end": 2,
                    },
                    {
                        "feature_name": "ecog",
                        "value": "3",
                        "evidence": "ECOG was 3",
                        "evidence_start": 0,
                        "evidence_end": 10,
                        "recorded_at": None,
                        "recorded_at_evidence": None,
                        "recorded_at_start": None,
                        "recorded_at_end": None,
                    },
                ],
            }
        ]
    }

    with pytest.raises(stage2_analysis._PageObservationValidationError) as error:
        stage2_analysis._validate_page_observations(
            response,
            page=page,
            definitions=[definition],
        )

    retained = error.value.response["rows"][0]["observations"]
    assert len(retained) == 1
    assert retained[0]["offset_resolution"] == "nearest_exact_match"
    assert retained[0]["recorded_at_offset_resolution"] == "nearest_exact_match"
    assert retained[0]["source_start"] == 100 + text.index("ECOG was 2")
    assert retained[0]["recorded_at_source_start"] == 100 + text.index("2024-06-03")
    assert "not an exact substring" in error.value.issues[0]["reason"]


def test_stage2_iterative_consolidation_does_not_lose_candidates():
    prompts = []

    def completion(messages, request_config):
        body = prompt_inputs(messages)
        prompts.append(
            (
                _prompt_job(body),
                sum(len(message["content"]) for message in messages),
                request_config.max_prompt_chars,
            )
        )
        assert "packet_1" not in messages[1]["content"]
        job = _prompt_job(body)
        if job == "consolidate_stage2_candidate_pool":
            names = [feature["name"] for feature in body["features"]]
            if names == ["performance_status"]:
                return json.dumps({"merge_directives": []})
            assert names == [f"performance_status_{index}" for index in range(1, 7)]
            return json.dumps(
                {
                    "merge_directives": [
                        {
                            "inputs": [f"performance_status_{index}" for index in range(1, 7)],
                            "output": "performance_status",
                        }
                    ],
                }
            )
        assert job == "operationalize_stage2_candidate_group"
        assert body["supporting_evidence"] == [
            "Original clinical evidence for the candidate feature."
        ]
        return json.dumps(
            {
                "description": "Baseline ECOG performance status.",
                "value_type": "ordinal",
                "categories_or_unit": ["ECOG 0", "ECOG 1", "ECOG 2"],
                "measurement_definition": "Extract the pretreatment ECOG score.",
                "missing_value_rule": "Return null when undocumented.",
                "stability_summary": "Supported across bounded batches.",
                "caveats": "none",
            }
        )

    config = PlainHandoffStage2Config(
        endpoint="http://stage2.test/v1",
        model="test-model",
        max_prompt_chars=8_000,
        consolidation_max_prompt_chars=50_000,
    )
    runner = PlainHandoffStage2(
        config=config,
        clinical_question="Identify confounders.",
        completion=completion,
    )
    candidates = [
        {
            "candidate_id": f"candidate_{index:04d}",
            "architecture": f"architecture_{index}",
            "name": f"performance_status_{index}",
            "description": "ECOG evidence " + ("x" * 1_200),
            "value_type": "ordinal",
            "supporting_packet_ids": [f"packet_{index}"],
            "evidence_axes": ["treatment", "outcome"],
            "caveats": "none",
        }
        for index in range(1, 7)
    ]

    result = runner._consolidate_candidates(
        outer_fold=1,
        candidates=candidates,
        evidence_packets=_original_evidence_packets(
            [candidate["supporting_packet_ids"][0] for candidate in candidates]
        ),
    )

    assert [job for job, _size, _limit in prompts] == [
        "consolidate_stage2_candidate_pool",
        "consolidate_stage2_candidate_pool",
        "operationalize_stage2_candidate_group",
    ]
    assert all(size <= limit for _job, size, limit in prompts)
    assert prompts[0][2] == config.consolidation_max_prompt_chars
    assert prompts[1][2] == config.consolidation_max_prompt_chars
    assert prompts[2][2] == config.operationalization_max_prompt_chars
    assert set(result["candidate_dispositions"]) == {
        candidate["candidate_id"] for candidate in candidates
    }
    assert set(result["features"][0]["supporting_packet_ids"]) == {
        candidate["supporting_packet_ids"][0] for candidate in candidates
    }


def test_consolidation_dispositions_follow_registry_origin_candidates():
    def completion(messages, _request_config):
        body = prompt_inputs(messages)
        if _prompt_job(body) == "consolidate_stage2_candidate_pool":
            return json.dumps({"merge_directives": []})
        return json.dumps(
            {
                "description": "Patient age.",
                "value_type": "continuous",
                "categories_or_unit": ["years"],
                "measurement_definition": "Extract pretreatment age in years.",
                "missing_value_rule": "Return null when undocumented.",
                "stability_summary": "Supported by the cited evidence.",
                "caveats": "",
            }
        )

    runner = PlainHandoffStage2(
        config=PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
        ),
        clinical_question="Identify confounders.",
        completion=completion,
    )
    result = runner._consolidate_candidates(
        outer_fold=1,
        candidates=[
            {
                "candidate_id": "candidate_registry_0001",
                "origin_candidate_ids": ["candidate_0001", "candidate_0002"],
                "architecture": "deterministic_candidate_group",
                "name": "patient_age",
                "description": "Patient age.",
                "supporting_packet_ids": ["packet_1"],
                "ontology_packet_ids": ["packet_1"],
                "evidence_axes": ["treatment", "outcome"],
            }
        ],
        evidence_packets=_original_evidence_packets(["packet_1"]),
    )

    assert set(result["candidate_dispositions"]) == {"candidate_0001", "candidate_0002"}
    assert result["candidate_dispositions"]["candidate_0001"]["status"] == "retained"
    assert result["candidate_dispositions"]["candidate_0002"]["status"] == "merged"


def test_global_candidate_pool_prompt_exposes_all_unique_names_and_descriptions():
    messages = stage2_workflow._global_candidate_pool_prompt(groups=[
        {"name": "patient_age", "description": "Patient age.", "member_measurements": [{"description": "Age expressed in years."}]},
        {"name": "age_2", "description": "A value-encoded age alias."}, {"name": "serum_sodium", "description": "Serum sodium."}])
    text = messages[1]["content"]
    for text_value in ("patient age", "Patient age.", "Age expressed in years.", "age 2", "serum sodium"):
        assert text_value in text
    assert "candidate_id" not in str(messages) and "group_id" not in str(messages)
    result = stage2_workflow._validate_global_candidate_pool_directives({"merges": [{"members": ["patient age", "age 2"], "canonical_label": "Patient age"}]}, group_names=["patient_age", "age_2", "serum_sodium"])
    assert result["merge_directives"] == [{"inputs": ["patient_age", "age_2"], "output": "patient_age"}]


def test_alphabetical_candidate_batches_shift_boundaries_between_rounds():
    groups = [
        {"candidate_id": f"candidate_{name}", "name": name}
        for name in reversed(["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf"])
    ]

    offsets_and_names = []
    for round_number in range(1, 4):
        offset, batches = stage2_workflow._alphabetical_candidate_batches(
            groups,
            batch_size=3,
            round_number=round_number,
        )
        offsets_and_names.append(
            (
                offset,
                [[str(group["name"]) for group in batch] for batch in batches],
            )
        )

    assert offsets_and_names == [
        (0, [["alpha", "bravo", "charlie"], ["delta", "echo", "foxtrot"], ["golf"]]),
        (1, [["alpha"], ["bravo", "charlie", "delta"], ["echo", "foxtrot", "golf"]]),
        (2, [["alpha", "bravo"], ["charlie", "delta", "echo"], ["foxtrot", "golf"]]),
    ]


def test_candidate_consolidation_batches_switch_to_reproducible_seeded_shuffles():
    groups = [
        {"candidate_id": f"candidate_{index}", "name": name}
        for index, name in enumerate(
            ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf"],
            start=1,
        )
    ]

    alphabetical = stage2_workflow._candidate_consolidation_batches(
        groups,
        batch_size=3,
        round_number=2,
        alphabetical_rounds=2,
        seed=91,
    )
    shuffled = stage2_workflow._candidate_consolidation_batches(
        groups,
        batch_size=3,
        round_number=3,
        alphabetical_rounds=2,
        seed=91,
    )
    repeated = stage2_workflow._candidate_consolidation_batches(
        list(reversed(groups)),
        batch_size=3,
        round_number=3,
        alphabetical_rounds=2,
        seed=91,
    )

    assert alphabetical[:3] == ("alphabetical_shift", 1, None)
    assert shuffled[:3] == ("seeded_shuffle", None, 1)
    shuffled_names = [[group["name"] for group in batch] for batch in shuffled[3]]
    repeated_names = [[group["name"] for group in batch] for batch in repeated[3]]
    assert shuffled_names == repeated_names
    assert sorted(name for batch in shuffled_names for name in batch) == sorted(
        group["name"] for group in groups
    )
    assert shuffled_names != [
        ["alpha", "bravo", "charlie"],
        ["delta", "echo", "foxtrot"],
        ["golf"],
    ]


def test_iterative_consolidation_finds_aliases_across_a_shifted_batch_boundary(
    tmp_path: Path,
):
    names = ["aaa", "bbb", "marker_level", "marker_status", "yyy", "zzz"]
    groups = stage2_workflow._materialize_exact_name_groups(
        [
            {
                "candidate_id": f"candidate_{index:04d}",
                "architecture": f"architecture_{index}",
                "name": name,
                "description": f"Description for {name}.",
                "value_type": "ambiguous",
                "supporting_packet_ids": [f"packet_{index:04d}"],
                "evidence_axes": ["outcome"],
                "caveats": "",
            }
            for index, name in enumerate(names, start=1)
        ]
    )
    prompt_features = []

    def completion(messages, _config):
        body = prompt_inputs(messages)
        batch_names = [feature["name"] for feature in body["features"]]
        prompt_features.append(body["features"])
        if {"marker_level", "marker_status"} <= set(batch_names):
            return json.dumps(
                {
                    "merge_directives": [
                        {
                            "inputs": ["marker_level", "marker_status"],
                            "output": "marker_level",
                        }
                    ],
                }
            )
        return json.dumps({"merge_directives": []})

    output_dir = tmp_path / "candidate_pool_consolidation"
    consolidated = PlainHandoffStage2(
        config=PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
            workers=1,
            consolidation_batch_size=3,
            consolidation_max_rounds=3,
        ),
        clinical_question="Not supplied to consolidation.",
        completion=completion,
    )._consolidate_candidate_pool(
        outer_fold=1,
        groups=groups,
        output_dir=output_dir,
    )

    assert [group["name"] for group in consolidated] == [
        "aaa",
        "bbb",
        "marker_level",
        "yyy",
        "zzz",
    ]
    marker = next(group for group in consolidated if group["name"] == "marker_level")
    assert marker["origin_candidate_ids"] == ["candidate_0003", "candidate_0004"]
    marker_views = [
        feature
        for batch in prompt_features
        for feature in batch
        if feature["name"] == "marker_level" and len(feature["descriptions"]) == 2
    ]
    assert marker_views[-1]["descriptions"] == [
        "Description for marker_level.",
        "Description for marker_status.",
    ]
    assert any(
        [feature["name"] for feature in batch] == ["bbb", "marker_level", "marker_status"]
        for batch in prompt_features
    )
    summaries = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(output_dir.glob("round_*/complete.json"))
    ]
    assert [summary["boundary_offset"] for summary in summaries] == [0, 1, 2]
    assert [summary["changed"] for summary in summaries] == [False, True, False]


def test_seeded_shuffle_round_can_merge_candidates_from_distant_alphabetical_batches(
    tmp_path: Path,
):
    names = [
        "alpha",
        "bravo",
        "charlie",
        "delta",
        "echo",
        "foxtrot",
        "golf",
        "hotel",
        "india",
    ]
    groups = stage2_workflow._materialize_exact_name_groups(
        [
            {
                "candidate_id": f"candidate_{index:04d}",
                "architecture": "architecture_alpha",
                "name": name,
                "description": f"Description for {name}.",
                "supporting_packet_ids": [f"packet_{index:04d}"],
                "evidence_axes": ["outcome"],
            }
            for index, name in enumerate(names, start=1)
        ]
    )
    seed = 17
    _, alphabetical_batches = stage2_workflow._alphabetical_candidate_batches(
        groups,
        batch_size=3,
        round_number=1,
    )
    alphabetical_owner = {
        str(group["name"]): batch_index
        for batch_index, batch in enumerate(alphabetical_batches)
        for group in batch
    }
    shuffled_batches = stage2_workflow._seeded_shuffle_candidate_batches(
        groups,
        batch_size=3,
        seed=seed + 1_000_003,
        shuffle_round=1,
    )
    alias_pair = next(
        (str(left["name"]), str(right["name"]))
        for batch in shuffled_batches
        for left_index, left in enumerate(batch)
        for right in batch[left_index + 1 :]
        if alphabetical_owner[str(left["name"])] != alphabetical_owner[str(right["name"])]
    )

    def completion(messages, _config):
        body = prompt_inputs(messages)
        batch_names = {feature["name"] for feature in body["features"]}
        if set(alias_pair) <= batch_names:
            return json.dumps(
                {
                    "merge_directives": [{"inputs": list(alias_pair), "output": alias_pair[0]}],
                }
            )
        return json.dumps({"merge_directives": []})

    output_dir = tmp_path / "candidate_pool_consolidation"
    consolidated = PlainHandoffStage2(
        config=PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
            workers=1,
            consolidation_batch_size=3,
            consolidation_alphabetical_rounds=1,
            consolidation_max_rounds=2,
        ),
        clinical_question="Not supplied to consolidation.",
        completion=completion,
    )._consolidate_candidate_pool(
        outer_fold=1,
        groups=groups,
        output_dir=output_dir,
        seed=seed,
    )

    assert len(consolidated) == len(groups) - 1
    summaries = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(output_dir.glob("round_*/complete.json"))
    ]
    assert [summary["ordering"] for summary in summaries] == [
        "alphabetical_shift",
        "seeded_shuffle",
    ]
    assert [summary["changed"] for summary in summaries] == [False, True]


def test_iterative_consolidation_retains_batch_and_explicit_feature_after_invalid_responses(
    tmp_path: Path,
    caplog,
):
    caplog.set_level("WARNING", logger=stage2_workflow.__name__)
    config = plain_stage2_config_from_mapping(
        {
            "endpoint": "http://stage2.test/v1",
            "model": "test-model",
            "workers": 1,
            "consolidation_batch_size": 20,
            "consolidation_max_rounds": 1,
            "explicit_features": [
                {
                    "name": "investigator_marker",
                    "description": "Investigator-specified marker.",
                    "value_type": "binary",
                    "categories_or_unit": ["absent", "present"],
                    "measurement_definition": "Extract the documented marker status.",
                    "missing_value_rule": "Return null when undocumented.",
                    "roles": ["effect_modifier"],
                }
            ],
        },
        default_workers=1,
    )
    assert config is not None
    explicit = config.explicit_features[0]
    groups = stage2_workflow._materialize_exact_name_groups(
        [
            {
                "candidate_id": "candidate_alpha",
                "architecture": "architecture_alpha",
                "name": "alpha_measurement",
                "description": "An ordinary candidate measurement.",
                "supporting_packet_ids": ["packet_alpha"],
                "evidence_axes": ["outcome"],
            },
            {
                "candidate_id": "configured_explicit_feature_0001",
                "architecture": stage2_workflow.CONFIGURED_EXPLICIT_FEATURE_ARCHITECTURE,
                "name": explicit.name,
                "description": explicit.description,
                "value_type": explicit.value_type,
                "supporting_packet_ids": [],
                "evidence_axes": [],
                "configured_feature_definitions": [explicit.as_definition()],
            },
        ]
    )
    calls = 0

    def completion(_messages, _config):
        nonlocal calls
        calls += 1
        return json.dumps(
            {
                "merge_directives": [
                    {
                        "inputs": ["alpha_measurement", "not_a_supplied_feature"],
                        "output": "alpha_measurement",
                    }
                ],
            }
        )

    output_dir = tmp_path / "candidate_pool_consolidation"
    consolidated = PlainHandoffStage2(
        config=config,
        clinical_question="Not supplied to consolidation.",
        completion=completion,
    )._consolidate_candidate_pool(
        outer_fold=1,
        groups=groups,
        output_dir=output_dir,
    )

    assert calls == 16
    assert [group["name"] for group in consolidated] == [
        "alpha_measurement",
        "investigator_marker",
    ]
    retained_explicit = consolidated[1]
    assert retained_explicit["configured_feature_definitions"] == [explicit.as_definition()]
    fallback = json.loads(
        (output_dir / "round_001" / "batch_001" / "fallback.json").read_text(encoding="utf-8")
    )
    assert fallback["status"] == "conservative_passthrough"
    assert fallback["retained_feature_names"] == [
        "alpha_measurement",
        "investigator_marker",
    ]
    complete = json.loads((output_dir / "complete.json").read_text(encoding="utf-8"))
    assert complete["stopped_reason"] == "single_batch_validation_fallback"
    assert complete["validation_fallback_batches"] == 1
    assert "retaining all 2 supplied features unchanged" in caplog.text


def test_iterative_consolidation_cannot_exclude_an_ordinary_candidate(tmp_path: Path):
    groups = stage2_workflow._materialize_exact_name_groups(
        [
            {
                "candidate_id": "candidate_age",
                "architecture": "architecture_alpha",
                "name": "age",
                "description": "Patient age in years.",
                "supporting_packet_ids": ["packet_age"],
                "evidence_axes": ["treatment", "outcome"],
            },
            {
                "candidate_id": "candidate_sodium",
                "architecture": "architecture_beta",
                "name": "serum_sodium",
                "description": "Serum sodium concentration.",
                "supporting_packet_ids": ["packet_sodium"],
                "evidence_axes": ["outcome"],
            },
        ]
    )
    calls = 0

    def completion(_messages, _config):
        nonlocal calls
        calls += 1
        return json.dumps(
            {
                "merge_directives": [],
                "exclude_feature_names": ["age"],
            }
        )

    output_dir = tmp_path / "candidate_pool_consolidation"
    consolidated = PlainHandoffStage2(
        config=PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
            workers=1,
            consolidation_max_rounds=1,
        ),
        clinical_question="Not supplied to consolidation.",
        completion=completion,
    )._consolidate_candidate_pool(
        outer_fold=1,
        groups=groups,
        output_dir=output_dir,
    )

    assert calls == 16
    assert [group["name"] for group in consolidated] == ["age", "serum_sodium"]
    fallback = json.loads(
        (output_dir / "round_001" / "batch_001" / "fallback.json").read_text(encoding="utf-8")
    )
    assert fallback["status"] == "conservative_passthrough"
    assert "merge-only; omit exclude_feature_names" in fallback["validation_error"]


def test_global_group_merge_validator_maps_names_and_rejects_ambiguous_routes():
    result = stage2_workflow._validate_global_candidate_pool_directives(
        {
            "merge_directives": [
                {
                    "inputs": ["Patient Age", "age_2"],
                    "output": "age_at_baseline",
                }
            ],
        },
        group_names=["patient_age", "age_2", "serum_sodium"],
    )

    assert result == {
        "merge_directives": [
            {
                "inputs": ["patient_age", "age_2"],
                "output": "age_at_baseline",
            }
        ],
    }

    with pytest.raises(ValueError, match="unknown or ambiguous feature"):
        stage2_workflow._validate_global_candidate_pool_directives(
            {
                "merge_directives": [{"inputs": ["patient_age", "missing_name"], "output": "age"}],
            },
            group_names=["patient_age", "age_2", "serum_sodium"],
        )

    with pytest.raises(ValueError, match="only one directive"):
        stage2_workflow._validate_global_candidate_pool_directives(
            {
                "merge_directives": [
                    {"inputs": ["patient_age", "age_2"], "output": "age"},
                    {
                        "inputs": ["age_2", "serum_sodium"],
                        "output": "other",
                    },
                ],
            },
            group_names=["patient_age", "age_2", "serum_sodium"],
        )


def test_global_group_merge_validator_accepts_reused_output_in_its_complete_input_family():
    result = stage2_workflow._validate_global_candidate_pool_directives(
        {
            "merge_directives": [
                {
                    "inputs": [
                        "pd_l1_expression",
                        "pdl1_expression_level",
                        "pd_l1_expression_level",
                        "pd_l1_expression_status",
                        "tps_score",
                    ],
                    "output": "pd_l1_expression_level",
                }
            ],
        },
        group_names=[
            "pd_l1_expression",
            "pdl1_expression_level",
            "pd_l1_expression_level",
            "pd_l1_expression_status",
            "tps_score",
            "serum_sodium",
        ],
    )

    assert result["merge_directives"] == [
        {
            "inputs": [
                "pd_l1_expression",
                "pdl1_expression_level",
                "pd_l1_expression_level",
                "pd_l1_expression_status",
                "tps_score",
            ],
            "output": "pd_l1_expression_level",
        }
    ]


def test_global_group_merge_validator_completes_supplied_output_collision_routes():
    result = stage2_workflow._validate_global_candidate_pool_directives(
        {
            "merge_directives": [
                {
                    "inputs": ["pd_l1_expression", "pdl1_expression_level"],
                    "output": "pd_l1_expression_level",
                }
            ],
        },
        group_names=[
            "pd_l1_expression",
            "pdl1_expression_level",
            "pd_l1_expression_level",
        ],
    )

    assert result["merge_directives"] == [
        {
            "inputs": [
                "pd_l1_expression",
                "pdl1_expression_level",
                "pd_l1_expression_level",
            ],
            "output": "pd_l1_expression_level",
        }
    ]

    with pytest.raises(
        ValueError,
        match=(
            "global merge input names may appear in only one directive; directive 2 repeats "
            "inputs already used by earlier directives"
        ),
    ):
        stage2_workflow._validate_global_candidate_pool_directives(
            {
                "merge_directives": [
                    {
                        "inputs": ["pd_l1_expression", "pdl1_expression_level"],
                        "output": "pd_l1_expression_level",
                    },
                    {
                        "inputs": ["pd_l1_expression_level", "pd_l1_expression_status"],
                        "output": "pd_l1_expression_status",
                    },
                ],
            },
            group_names=[
                "pd_l1_expression",
                "pdl1_expression_level",
                "pd_l1_expression_level",
                "pd_l1_expression_status",
            ],
        )

    with pytest.raises(ValueError, match="merge-only; omit exclude_feature_names"):
        stage2_workflow._validate_global_candidate_pool_directives(
            {
                "merge_directives": [
                    {
                        "inputs": ["pd_l1_expression", "pdl1_expression_level"],
                        "output": "pd_l1_expression_level",
                    }
                ],
                "exclude_feature_names": ["pd_l1_expression_level"],
            },
            group_names=[
                "pd_l1_expression",
                "pdl1_expression_level",
                "pd_l1_expression_level",
            ],
        )


def test_global_group_merge_validator_completes_omitted_reused_output_input():
    result = stage2_workflow._validate_global_candidate_pool_directives(
        {
            "merge_directives": [
                {
                    "inputs": ["pd_l1_expression"],
                    "output": "pd_l1_expression_level",
                }
            ],
        },
        group_names=["pd_l1_expression", "pd_l1_expression_level"],
    )

    assert result["merge_directives"] == [
        {
            "inputs": ["pd_l1_expression", "pd_l1_expression_level"],
            "output": "pd_l1_expression_level",
        }
    ]


def test_global_group_merge_validator_resolves_supplied_descriptions_and_drops_noop():
    result = stage2_workflow._validate_global_candidate_pool_directives(
        {
            "merge_directives": [
                {
                    "inputs": [
                        "white blood cell count",
                        "The white blood cell (WBC) count.",
                    ],
                    "output": "white_blood_cell_count",
                }
            ],
        },
        group_names=["white_blood_cell_count", "worsening_symptoms"],
        group_descriptions={
            "white_blood_cell_count": [
                "white blood cell count",
                "The white blood cell (WBC) count.",
            ],
            "worsening_symptoms": ["worsening symptoms"],
        },
    )

    assert result == {"merge_directives": []}


def test_global_group_merge_validator_rejects_filtering_and_protects_configured_features():
    with pytest.raises(ValueError, match="merge-only; omit exclude_feature_names"):
        stage2_workflow._validate_global_candidate_pool_directives(
            {
                "merge_directives": [{"inputs": ["patient_age", "age_2"], "output": "patient_age"}],
                "exclude_feature_names": ["age_2"],
            },
            group_names=["patient_age", "age_2", "serum_sodium"],
        )

    with pytest.raises(ValueError, match="must not combine distinct investigator-configured"):
        stage2_workflow._validate_global_candidate_pool_directives(
            {
                "merge_directives": [
                    {
                        "inputs": ["patient_age", "serum_sodium"],
                        "output": "patient_age",
                    }
                ],
            },
            group_names=["patient_age", "serum_sodium"],
            configured_feature_names=["patient_age", "serum_sodium"],
        )


def test_global_group_merge_directives_merge_inputs_and_pass_through_others():
    groups = [
        {
            "candidate_id": "candidate_pool_group_0001",
            "architecture": "deterministic_candidate_group",
            "supporting_architectures": ["architecture_alpha"],
            "name": "blood_glucose_concentration",
            "description": "Pretreatment blood glucose concentration.",
            "value_type": "continuous",
            "supporting_packet_ids": ["packet_alpha"],
            "evidence_axes": ["outcome", "treatment"],
            "origin_candidate_ids": ["candidate_0001"],
            "member_measurements": [],
        },
        {
            "candidate_id": "candidate_pool_group_0002",
            "architecture": "deterministic_candidate_group",
            "supporting_architectures": ["architecture_beta"],
            "name": "glycemia",
            "description": "Pretreatment glycemia.",
            "value_type": "continuous",
            "supporting_packet_ids": ["packet_beta"],
            "evidence_axes": ["residual_effect"],
            "origin_candidate_ids": ["candidate_0002"],
            "member_measurements": [],
        },
        {
            "candidate_id": "candidate_pool_group_0003",
            "architecture": "deterministic_candidate_group",
            "supporting_architectures": ["architecture_gamma"],
            "name": "heart_rate",
            "description": "Pretreatment heart rate.",
            "value_type": "continuous",
            "supporting_packet_ids": ["packet_gamma"],
            "evidence_axes": ["outcome"],
            "origin_candidate_ids": ["candidate_0003"],
            "member_measurements": [],
        },
    ]

    merged = stage2_workflow._apply_global_candidate_pool_directives(
        groups,
        [
            {
                "inputs": ["blood_glucose_concentration", "glycemia"],
                "output": "blood_glucose_concentration",
            }
        ],
    )

    assert [group["name"] for group in merged] == [
        "blood_glucose_concentration",
        "heart_rate",
    ]
    assert merged[0]["origin_candidate_ids"] == ["candidate_0001", "candidate_0002"]
    assert merged[0]["supporting_packet_ids"] == ["packet_alpha", "packet_beta"]
    assert merged[0]["evidence_axes"] == ["outcome", "residual_effect", "treatment"]
    assert merged[1] == groups[2]


def test_redesigned_consolidation_assembles_provenance_roles_and_dispositions_in_python(
    tmp_path: Path,
):
    prompt_bodies = []
    packet_ids = [
        "packet_alpha_long_id",
        "packet_beta_long_id",
        "packet_gamma_long_id",
        "packet_delta_long_id",
    ]

    def completion(messages, _config):
        rendered = messages[1]["content"]
        assert all(packet_id not in rendered for packet_id in packet_ids)
        body = prompt_inputs(messages)
        prompt_bodies.append(body)
        job = _prompt_job(body)
        if job == "consolidate_stage2_candidate_pool":
            assert rendered.startswith("Variables\n")
            assert all("candidate" not in feature["name"] for feature in body["features"])
            names = {feature["name"] for feature in body["features"]}
            if not {"serum_sodium", "blood_sodium_concentration"} <= names:
                return json.dumps({"merge_directives": []})
            return json.dumps(
                {
                    "merge_directives": [
                        {
                            "inputs": [
                                "serum_sodium",
                                "blood_sodium_concentration",
                            ],
                            "output": "serum_sodium",
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
                "missing_value_rule": "Return null when no value is documented.",
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
            "name": "serum_sodium",
            "description": "Pretreatment serum sodium concentration.",
            "value_type": "continuous",
            "supporting_packet_ids": [packet_ids[0]],
            "evidence_axes": ["treatment", "outcome"],
            "caveats": "",
        },
        {
            "candidate_id": "candidate_0002",
            "architecture": "architecture_beta",
            "name": "body_mass_index",
            "description": "Pretreatment body mass index.",
            "value_type": "continuous",
            "supporting_packet_ids": [packet_ids[1]],
            "evidence_axes": ["outcome", "residual_effect"],
            "caveats": "",
        },
        {
            "candidate_id": "candidate_0003",
            "architecture": "architecture_gamma",
            "name": "heart_rate",
            "description": "Pretreatment resting heart rate.",
            "value_type": "continuous",
            "supporting_packet_ids": [packet_ids[2]],
            "evidence_axes": ["outcome"],
            "caveats": "",
        },
        {
            "candidate_id": "candidate_0004",
            "architecture": "architecture_delta",
            "name": "blood_sodium_concentration",
            "description": "Pretreatment sodium concentration in blood.",
            "value_type": "continuous",
            "supporting_packet_ids": [packet_ids[3]],
            "evidence_axes": ["treatment", "outcome"],
            "caveats": "",
        },
    ]

    checkpoint_dir = tmp_path / "consolidation"
    result = runner._consolidate_candidates(
        outer_fold=1,
        candidates=candidates,
        evidence_packets=_original_evidence_packets(packet_ids),
        output_dir=checkpoint_dir,
    )

    assert [feature["supporting_packet_ids"] for feature in result["features"]] == [
        [packet_ids[1]],
        [packet_ids[2]],
        [packet_ids[0], packet_ids[3]],
    ]
    assert [feature["supporting_architectures"] for feature in result["features"]] == [
        ["architecture_beta"],
        ["architecture_gamma"],
        ["architecture_alpha", "architecture_delta"],
    ]
    assert [feature["roles"] for feature in result["features"]] == [[], [], []]
    assert result["candidate_dispositions"]["candidate_0001"]["status"] == "retained"
    assert result["candidate_dispositions"]["candidate_0002"]["status"] == "retained"
    assert result["candidate_dispositions"]["candidate_0003"]["status"] == "retained"
    assert result["candidate_dispositions"]["candidate_0004"]["status"] == "merged"
    assert {_prompt_job(body) for body in prompt_bodies} == {
        "consolidate_stage2_candidate_pool",
        "operationalize_stage2_candidate_group",
    }
    global_input = json.loads(
        (checkpoint_dir / "candidate_pool_consolidation" / "input.json").read_text(encoding="utf-8")
    )
    assert "clinical_question" not in global_input
    assert len(list(checkpoint_dir.rglob("complete.json"))) == 8

    def unexpected_completion(_messages, _config):
        raise AssertionError("completed consolidation leaves must resume from checkpoints")

    resumed = PlainHandoffStage2(
        config=runner.config,
        clinical_question=runner.clinical_question,
        completion=unexpected_completion,
    )._consolidate_candidates(
        outer_fold=1,
        candidates=candidates,
        evidence_packets=_original_evidence_packets(packet_ids),
        output_dir=checkpoint_dir,
    )
    assert resumed == result


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


def test_shifted_consolidation_round_preserves_explicit_feature_name_ontology_and_roles():
    config = plain_stage2_config_from_mapping(
        {
            "endpoint": "http://stage2.test/v1",
            "model": "test-model",
            "workers": 1,
            "consolidation_batch_size": 2,
            "consolidation_max_rounds": 2,
            "explicit_features": [
                {
                    "name": "ecog_performance_status",
                    "description": "Investigator-specified ECOG performance status.",
                    "value_type": "ordinal",
                    "categories_or_unit": ["0", "1", "2", "3", "4"],
                    "measurement_definition": "Extract the explicitly documented ECOG score.",
                    "missing_value_rule": "Return null when ECOG is undocumented.",
                    "roles": ["effect_modifier"],
                }
            ],
        },
        default_workers=1,
    )
    assert config is not None
    explicit = config.explicit_features[0]
    raw_candidates = [
        {
            "candidate_id": "candidate_aaa",
            "architecture": "architecture_alpha",
            "name": "aaa",
            "description": "Unrelated scalar A.",
            "supporting_packet_ids": ["packet_aaa"],
            "evidence_axes": ["outcome"],
        },
        {
            "candidate_id": "configured_explicit_feature_0001",
            "architecture": stage2_workflow.CONFIGURED_EXPLICIT_FEATURE_ARCHITECTURE,
            "name": explicit.name,
            "description": explicit.description,
            "value_type": explicit.value_type,
            "supporting_packet_ids": [],
            "evidence_axes": [],
            "configured_feature_definitions": [explicit.as_definition()],
        },
        {
            "candidate_id": "candidate_ecog_alias",
            "architecture": "architecture_beta",
            "name": "ecog_status",
            "description": "A discovered alias of ECOG performance status.",
            "supporting_packet_ids": ["packet_ecog"],
            "evidence_axes": ["treatment", "outcome"],
        },
        {
            "candidate_id": "candidate_zzz",
            "architecture": "architecture_gamma",
            "name": "zzz",
            "description": "Unrelated scalar Z.",
            "supporting_packet_ids": ["packet_zzz"],
            "evidence_axes": ["outcome"],
        },
    ]
    groups = stage2_workflow._materialize_exact_name_groups(raw_candidates)
    configured_batches = []

    def completion(messages, _config):
        body = prompt_inputs(messages)
        names = [feature["name"] for feature in body["features"]]
        if "ecog_performance_status" in names:
            configured_batches.append(body["configured_feature_names"])
        if {"ecog_performance_status", "ecog_status"} <= set(names):
            return json.dumps(
                {
                    "merge_directives": [
                        {
                            "inputs": ["ecog_performance_status", "ecog_status"],
                            "output": "ecog_performance_status",
                        }
                    ],
                }
            )
        return json.dumps({"merge_directives": []})

    consolidated = PlainHandoffStage2(
        config=config,
        clinical_question="Not supplied to consolidation.",
        completion=completion,
    )._consolidate_candidate_pool(outer_fold=1, groups=groups)

    assert configured_batches == [
        ["ecog_performance_status"],
        ["ecog_performance_status"],
    ]
    explicit_group = next(
        group for group in consolidated if group["name"] == "ecog_performance_status"
    )
    assert explicit_group["origin_candidate_ids"] == [
        "configured_explicit_feature_0001",
        "candidate_ecog_alias",
    ]
    assert explicit_group["configured_feature_definitions"] == [explicit.as_definition()]
    assert stage2_workflow._group_roles(explicit_group) == ["effect_modifier"]


def test_configured_feature_is_retained_without_any_discovered_candidate():
    config = plain_stage2_config_from_mapping(
        {
            "endpoint": "http://stage2.test/v1",
            "model": "test-model",
            "explicit_features": {
                "features": [
                    {
                        "name": "age_at_treatment_decision",
                        "description": "Age at the pretreatment decision point.",
                        "type": "continuous",
                        "unit": "years",
                        "measurement_definition": (
                            "Extract age in years at the treatment decision point."
                        ),
                        "missing_value_rule": "Return null when age cannot be determined.",
                        "roles": ["confounder"],
                    }
                ]
            },
        },
        default_workers=1,
    )
    assert config is not None

    runner = PlainHandoffStage2(
        config=config,
        clinical_question="Estimate the treatment effect.",
        completion=lambda _messages, _config: (_ for _ in ()).throw(
            AssertionError("a configured-only feature must not request ontology definition")
        ),
    )
    result = runner._consolidate_candidates(
        outer_fold=1,
        candidates=[],
        evidence_packets=[],
    )

    assert [feature["name"] for feature in result["features"]] == ["age_at_treatment_decision"]
    assert result["features"][0]["categories_or_unit"] == ["years"]
    assert result["features"][0]["roles"] == ["confounder"]


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


def test_iterative_batch_jointly_merges_general_threshold_value_and_score_representations():
    names = [
        "inflammation_marker_expression",
        "high_inflammation_marker_expression",
        "inflammation_marker_30_percent",
        "inflammation_marker_assay_score",
    ]
    packet_ids = [f"packet_{index}" for index in range(1, 5)]
    evidence_packets = [
        {
            "packet_id": packet_id,
            "content": {
                "representative_evidence": [
                    {"text": f"Evidence representation {index} of the same marker assay."}
                ]
            },
        }
        for index, packet_id in enumerate(packet_ids, start=1)
    ]
    jobs = []

    def completion(messages, _config):
        body = prompt_inputs(messages)
        job = _prompt_job(body)
        jobs.append(job)
        if job == "consolidate_stage2_candidate_pool":
            prompt_names = [feature["name"] for feature in body["features"]]
            if prompt_names == ["inflammation_marker_expression"]:
                return json.dumps({"merge_directives": []})
            assert prompt_names == sorted(names)
            return json.dumps(
                {
                    "merge_directives": [
                        {
                            "inputs": names,
                            "output": "inflammation_marker_expression",
                        }
                    ],
                }
            )
        assert job == "operationalize_stage2_candidate_group"
        assert body["candidate_feature_name"] == "inflammation_marker_expression"
        return json.dumps(
            {
                "description": "Quantitative inflammation-marker assay result.",
                "value_type": "continuous",
                "categories_or_unit": ["%"],
                "measurement_definition": "Extract the documented assay percentage.",
                "missing_value_rule": "Return null when the assay is undocumented.",
                "stability_summary": "One assay represented at several granularities.",
                "caveats": "",
            }
        )

    candidates = [
        {
            "candidate_id": f"candidate_{index:04d}",
            "architecture": f"architecture_{index}",
            "name": name,
            "description": description,
            "value_type": "ambiguous",
            "supporting_packet_ids": [packet_id],
            "evidence_axes": axes,
            "caveats": "",
        }
        for index, (name, description, packet_id, axes) in enumerate(
            zip(
                names,
                [
                    "The marker expression measurement.",
                    "A high thresholded state of the marker expression measurement.",
                    "One observed 30 percent value of the marker expression measurement.",
                    "The quantitative assay score for the marker expression measurement.",
                ],
                packet_ids,
                [
                    ["outcome"],
                    ["residual_effect"],
                    ["treatment"],
                    ["outcome"],
                ],
            ),
            start=1,
        )
    ]
    result = PlainHandoffStage2(
        config=PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
        ),
        clinical_question="Intentionally absent from consolidation.",
        completion=completion,
    )._consolidate_candidates(
        outer_fold=1,
        candidates=candidates,
        evidence_packets=evidence_packets,
    )

    assert jobs == [
        "consolidate_stage2_candidate_pool",
        "consolidate_stage2_candidate_pool",
        "operationalize_stage2_candidate_group",
    ]
    assert [feature["name"] for feature in result["features"]] == ["inflammation_marker_expression"]
    assert result["features"][0]["supporting_packet_ids"] == packet_ids
    assert result["features"][0]["roles"] == []
    assert [
        result["candidate_dispositions"][f"candidate_{index:04d}"]["status"]
        for index in range(1, 5)
    ] == ["retained", "merged", "merged", "merged"]


def test_operationalization_ignores_model_authored_provenance_and_uses_aliases():
    result = stage2_workflow._validate_operationalization(
        {
            "feature": {
                "name": "model_renamed_measurement",
                "description": "A scalar measurement.",
                "data_type": "numeric",
                "unit": "standard unit",
                "operational_definition": "Extract the pretreatment value.",
                "missingness_rule": "Return null when undocumented.",
                "supporting_packet_ids": ["mistyped_packet_id"],
                "roles": ["confounder"],
            },
        },
        group={
            "name": "scalar_measurement",
            "description": "A scalar measurement.",
            "value_type": "continuous",
            "supporting_packet_ids": ["real_packet_id"],
            "supporting_architectures": ["real_architecture"],
        },
    )

    assert result["value_type"] == "continuous"
    assert result["categories_or_unit"] == ["standard unit"]
    assert result["measurement_definition"] == "Extract the pretreatment value."
    assert result["missing_value_rule"] == "Return null when undocumented."
    assert result["conflict_resolution"] == {
        "strategy": "latest",
        "positive_category": None,
    }
    assert "name" not in result
    assert "supporting_packet_ids" not in result
    assert "roles" not in result


def test_operationalization_supplies_safe_defaults_for_omitted_leaf_fields():
    result = stage2_workflow._validate_operationalization(
        {
            "description": "Pretreatment scalar measurement.",
            "value_type": "continuous",
            "categories_or_unit": ["standard unit"],
        },
        group={
            "name": "scalar_measurement",
            "description": "Pretreatment scalar measurement.",
            "value_type": "continuous",
            "supporting_packet_ids": ["packet_1", "packet_2"],
            "supporting_architectures": ["architecture_1"],
        },
    )

    assert result["measurement_definition"].startswith(
        "Extract one pretreatment scalar for scalar measurement"
    )
    assert result["missing_value_rule"].startswith("Return null")
    assert result["conflict_resolution"]["strategy"] == "single_or_null"
    assert result["stability_summary"] == (
        "Supported by 2 evidence packet(s) across 1 Stage 1 architecture(s)."
    )


def test_operationalization_requires_model_authored_value_type():
    with pytest.raises(ValueError, match="requires the model to choose value_type"):
        stage2_workflow._validate_operationalization(
            {
                "description": "A scalar measurement.",
                "categories_or_unit": ["standard unit"],
                "measurement_definition": "Extract the documented value.",
                "missing_value_rule": "Return null when undocumented.",
            },
            group={
                "name": "scalar_measurement",
                "value_type": "continuous",
                "supporting_packet_ids": ["packet_1"],
            },
        )


def test_operationalization_validates_structured_conflict_resolution():
    with pytest.raises(ValueError, match="requires a continuous feature"):
        stage2_workflow._validate_operationalization(
            {
                "description": "A binary status.",
                "value_type": "binary",
                "categories_or_unit": ["Absent", "Present"],
                "measurement_definition": "Extract the documented status.",
                "missing_value_rule": "Return null when undocumented.",
                "conflict_resolution": {
                    "strategy": "maximum",
                    "positive_category": None,
                },
            },
            group={
                "name": "binary_status",
                "supporting_packet_ids": ["packet_1"],
                "supporting_architectures": ["architecture_1"],
            },
        )

    result = stage2_workflow._validate_operationalization(
        {
            "description": "Whether the condition was ever documented.",
            "value_type": "binary",
            "categories_or_unit": ["Absent, Present"],
            "measurement_definition": "Extract whether there is any history of the condition.",
            "missing_value_rule": "Return null when documentation is insufficient.",
            "conflict_resolution": {
                "strategy": "any_positive",
                "positive_category": "Present",
            },
        },
        group={
            "name": "condition_history",
            "supporting_packet_ids": ["packet_1"],
            "supporting_architectures": ["architecture_1"],
        },
    )
    assert result["categories_or_unit"] == ["Absent", "Present"]
    assert result["conflict_resolution"] == {
        "strategy": "any_positive",
        "positive_category": "Present",
    }


@pytest.mark.parametrize(
    ("value_type", "categories", "message"),
    [
        ("binary", ["binary"], "exactly two distinct"),
        ("binary", ["enabled or disabled"], "exactly two distinct"),
        ("binary", ["disabled", "enabled", "unknown"], "exactly two distinct"),
        ("categorical", ["only_state"], "at least two distinct"),
        ("ordinal", ["single_level"], "at least two distinct"),
        ("binary", ["Disabled", "disabled"], "distinct after case and spacing"),
        ("binary", ["binary", "disabled"], "schema label"),
    ],
)
def test_operationalization_rejects_malformed_closed_ontologies(value_type, categories, message):
    with pytest.raises(ValueError, match=message):
        stage2_workflow._validate_operationalization(
            {
                "description": "A generic scalar state.",
                "value_type": value_type,
                "categories_or_unit": categories,
                "measurement_definition": "Extract the documented pretreatment state.",
                "missing_value_rule": "Return null when undocumented.",
            },
            group={
                "name": "generic_state",
                "description": "A generic scalar state.",
                "value_type": value_type,
                "supporting_packet_ids": ["packet_1"],
                "supporting_architectures": ["architecture_1"],
            },
        )


def test_operationalization_retries_malformed_ontology_then_accepts_repair():
    calls = []

    def completion(messages, _config):
        calls.append(messages)
        categories = ["binary"] if len(calls) == 1 else ["disabled", "enabled"]
        return json.dumps(
            {
                "description": "A generic scalar state.",
                "value_type": "binary",
                "categories_or_unit": categories,
                "measurement_definition": "Extract the documented pretreatment state.",
                "missing_value_rule": "Return null when undocumented.",
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
    result = runner._operationalize_candidate_group(
        group={
            "candidate_id": "group_001",
            "name": "generic_state",
            "description": "A generic scalar state.",
            "value_type": "binary",
            "evidence_axes": ["outcome"],
            "supporting_packet_ids": ["packet_1"],
            "supporting_architectures": ["architecture_1"],
            "ontology_packet_ids": ["packet_1"],
        },
        packet_by_id={
            packet["packet_id"]: packet for packet in _original_evidence_packets(["packet_1"])
        },
    )

    assert result["categories_or_unit"] == ["disabled", "enabled"]
    assert len(calls) == 2
    assert "exactly two distinct" in calls[1][-1]["content"]


def test_operationalization_duplicate_category_repair_names_colliding_values():
    calls = []

    def completion(messages, _config):
        calls.append(messages)
        categories = ["Disabled", "disabled"] if len(calls) == 1 else ["disabled", "enabled"]
        return json.dumps(
            {
                "description": "A generic scalar state.",
                "value_type": "binary",
                "categories_or_unit": categories,
                "measurement_definition": "Extract the documented pretreatment state.",
                "missing_value_rule": "Return null when undocumented.",
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
    result = runner._operationalize_candidate_group(
        group={
            "candidate_id": "group_001",
            "name": "generic_state",
            "description": "A generic scalar state.",
            "value_type": "binary",
            "evidence_axes": ["outcome"],
            "supporting_packet_ids": ["packet_1"],
            "supporting_architectures": ["architecture_1"],
            "ontology_packet_ids": ["packet_1"],
        },
        packet_by_id={
            packet["packet_id"]: packet for packet in _original_evidence_packets(["packet_1"])
        },
    )

    assert result["categories_or_unit"] == ["disabled", "enabled"]
    assert len(calls) == 2
    repair = calls[1][-1]["content"]
    assert "return each category once" in repair
    assert "Disabled" in repair
    assert "disabled" in repair


def test_operationalization_packs_oversized_supporting_evidence_under_independent_budget(
    tmp_path: Path,
):
    packet_ids = [f"packet_{index:03d}" for index in range(12)]
    packet_by_id = {
        packet_id: {
            "packet_id": packet_id,
            "content": {
                "representative_evidence": [
                    {
                        "text": (
                            f"Distinct clinical evidence excerpt {index}. "
                            + (chr(ord("a") + index) * 2_500)
                        )
                    }
                ]
            },
        }
        for index, packet_id in enumerate(packet_ids)
    }
    observed: dict[str, object] = {}

    def completion(messages, request_config):
        observed["prompt_chars"] = sum(len(message["content"]) for message in messages)
        observed["prompt_limit"] = request_config.max_prompt_chars
        observed["body"] = prompt_inputs(messages)
        return json.dumps(
            {
                "description": "A generic quantitative clinical measurement.",
                "value_type": "continuous",
                "categories_or_unit": ["standard unit"],
                "measurement_definition": "Extract the documented pretreatment value.",
                "missing_value_rule": "Return null when undocumented.",
            }
        )

    output_dir = tmp_path / "operationalization"
    result = PlainHandoffStage2(
        config=PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
            max_prompt_chars=4_000,
            operationalization_max_prompt_chars=8_000,
        ),
        clinical_question="Estimate a treatment effect.",
        completion=completion,
    )._operationalize_candidate_group(
        group={
            "candidate_id": "group_001",
            "name": "generic_quantitative_measurement",
            "description": "A generic quantitative clinical measurement.",
            "evidence_axes": ["outcome"],
            "supporting_packet_ids": packet_ids,
            "supporting_architectures": ["architecture_1"],
            "ontology_packet_ids": packet_ids,
        },
        packet_by_id=packet_by_id,
        output_dir=output_dir,
    )

    assert result["value_type"] == "continuous"
    assert observed["prompt_limit"] == 8_000
    assert int(observed["prompt_chars"]) <= 8_000
    body = observed["body"]
    assert isinstance(body, dict)
    assert 0 < len(body["supporting_evidence"]) < len(packet_ids)
    checkpoint_input = json.loads((output_dir / "input.json").read_text(encoding="utf-8"))
    packing = checkpoint_input["evidence_packing"]
    assert packing["available_evidence_items"] == len(packet_ids)
    assert packing["included_evidence_items"] == len(body["supporting_evidence"])
    assert packing["omitted_evidence_items"] > 0
    assert packing["prompt_chars"] == observed["prompt_chars"]
    assert packing["initial_prompt_budget_chars"] < packing["request_prompt_limit_chars"]


def test_operationalization_uses_audited_ambiguous_fallback_after_exhausted_repairs(
    tmp_path: Path,
):
    calls = 0

    def completion(_messages, _config):
        nonlocal calls
        calls += 1
        return json.dumps(
            {
                "description": "A generic clinical state.",
                "value_type": "binary",
                "categories_or_unit": ["low-intermediate"],
                "measurement_definition": "Extract the documented pretreatment state.",
                "missing_value_rule": "Return null when undocumented.",
            }
        )

    packet_by_id = {
        packet["packet_id"]: packet for packet in _original_evidence_packets(["packet_1"])
    }
    group = {
        "candidate_id": "group_001",
        "name": "generic_clinical_state",
        "description": "A generic clinical state.",
        "evidence_axes": ["outcome"],
        "supporting_packet_ids": ["packet_1"],
        "supporting_architectures": ["architecture_1"],
        "ontology_packet_ids": ["packet_1"],
    }
    output_dir = tmp_path / "operationalization"
    runner = PlainHandoffStage2(
        config=PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
            operationalization_max_prompt_chars=20_000,
        ),
        clinical_question="Estimate a treatment effect.",
        completion=completion,
    )

    first = runner._operationalize_candidate_group(
        group=group,
        packet_by_id=packet_by_id,
        output_dir=output_dir,
    )
    second = runner._operationalize_candidate_group(
        group=group,
        packet_by_id=packet_by_id,
        output_dir=output_dir,
    )

    assert calls == 16
    assert first == second
    assert first["value_type"] == "ambiguous"
    assert first["categories_or_unit"] == []
    assert "conservatively marked ambiguous" in first["caveats"]
    fallback = json.loads((output_dir / "fallback.json").read_text(encoding="utf-8"))
    assert fallback["status"] == "conservative_validation_fallback"
    assert "exactly two distinct" in fallback["validation_error"]
    complete = json.loads((output_dir / "complete.json").read_text(encoding="utf-8"))
    assert complete["status"] == "complete_with_validation_fallback"
    assert complete["validation_fallback"] is True


def test_stage2_continuous_nuisances_predict_with_and_without_covariates():
    treatment = np.tile([0, 1], 40)
    design = np.linspace(-2.0, 2.0, len(treatment)).reshape(-1, 1)
    outcome = 1.0 + 2.0 * design[:, 0] + 3.0 * treatment
    heldout = np.array([[-1.0], [0.3], [1.2]])

    models = stage2_analysis._fit_outcome_models(
        design, treatment, outcome, binary=False, seed=13
    )
    control, treated = stage2_analysis._predict_outcomes(models, heldout)

    assert models.control.__class__.__name__ == "ElasticNetRegressor"
    assert models.treated.__class__.__name__ == "ElasticNetRegressor"
    np.testing.assert_allclose(control, 1.0 + 2.0 * heldout[:, 0], atol=0.05)
    np.testing.assert_allclose(treated, 4.0 + 2.0 * heldout[:, 0], atol=0.05)

    intercept_only = stage2_analysis._fit_outcome_models(
        np.empty((len(treatment), 0)), treatment, outcome, binary=False, seed=13
    )
    control, treated = stage2_analysis._predict_outcomes(
        intercept_only, np.empty((len(heldout), 0))
    )
    np.testing.assert_allclose(control, outcome[treatment == 0].mean())
    np.testing.assert_allclose(treated, outcome[treatment == 1].mean())


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


def test_harmonization_plan_conservatively_normalizes_value_map_bookkeeping():
    feature = {"feature_id": "feature_assay", "name": "assay_measurement"}
    observations = {
        "numeric_count": 2,
        "numeric_quantiles": [
            {"probability": 0.0, "value": 1.0},
            {"probability": 1.0, "value": 2.0},
        ],
        "categorical_count": 3,
        "categorical_values": [
            {"raw_value": "reported one", "count": 1},
            {"raw_value": "reported two", "count": 1},
            {"raw_value": "reported three", "count": 1},
        ],
    }

    plan = stage2_analysis._validate_harmonization_plan(
        {
            "target_representation": "continuous",
            "reason": "Each usable text token denotes an exact assay value.",
            "canonical_categories": [],
            "categorical_value_map": [
                None,
                {"raw_value": "", "canonical_value": 9.0},
                {"raw_value": "unobserved", "canonical_value": 8.0},
                {"raw_value": "reported one", "canonical_value": 1.0},
                {"raw_value": "reported one", "canonical_value": 1.0},
                {"raw_value": "reported two", "canonical_value": 2.0},
                {"raw_value": "reported two", "canonical_value": 3.0},
            ],
            "numeric_bin_rules": [],
        },
        feature=feature,
        observations=observations,
    )

    mapping = {
        row["raw_value"]: row["canonical_value"]
        for row in plan["categorical_value_map"]
    }
    assert mapping == {
        "reported one": 1.0,
        "reported two": None,
        "reported three": None,
    }
    normalization = plan["categorical_value_map_normalization"]
    assert normalization["non_object_entries_dropped"] == 1
    assert normalization["empty_raw_value_entries_dropped"] == 1
    assert normalization["extra_raw_values_dropped"] == ["unobserved"]
    assert normalization["identical_duplicate_raw_values_deduplicated"] == [
        "reported one"
    ]
    assert normalization["conflicting_duplicate_raw_values_mapped_to_null"] == [
        "reported two"
    ]
    assert normalization["missing_raw_values_mapped_to_null"] == ["reported three"]


def _generic_feature_with_harmonization_plan():
    return {
        "feature_id": "feature_assay",
        "name": "assay_measurement",
        "description": "A pretreatment assay measurement.",
        "value_type": "continuous",
        "categories_or_unit": ["units"],
        "modeling_strategy": "categorical",
        "roles": ["prognostic"],
        "measurement_definition": "Extract the documented pretreatment assay measurement.",
        "missing_value_rule": "Return null when undocumented.",
        "harmonization_plan": {
            "schema_version": "stage2_mixed_value_harmonization_v1_llm_training_only",
            "feature_id": "feature_assay",
            "target_representation": "categorical",
            "reason": "Text labels and numeric values share two ordered tiers.",
            "canonical_categories": ["lower", "upper"],
            "categorical_value_map": [
                {"raw_value": "lower label", "canonical_value": "lower"},
                {"raw_value": "upper label", "canonical_value": "upper"},
            ],
            "numeric_bin_rules": [
                {
                    "lower_bound": None,
                    "lower_inclusive": False,
                    "upper_bound": 50,
                    "upper_inclusive": False,
                    "canonical_value": "lower",
                },
                {
                    "lower_bound": 50,
                    "lower_inclusive": True,
                    "upper_bound": None,
                    "upper_inclusive": False,
                    "canonical_value": "upper",
                },
            ],
            "unmapped_value_rule": "null",
            "training_observations_fingerprint": "prior",
        },
    }


def test_harmonization_extends_only_new_values_in_a_frozen_prior_plan(tmp_path: Path):
    extracted = pd.DataFrame(
        {
            "_oci_row_id": list(range(5)),
            "assay_measurement": [10.0, 75.0, "lower label", "upper label", "new label"],
        }
    )
    jobs = []

    def request_json(messages, validate, *, request_kind="interpretation"):
        assert request_kind == "interpretation"
        body = prompt_inputs(messages)
        jobs.append(body["job"])
        assert body["new_observed_training_text_values"] == [
            {"raw_value": "new label", "count": 1}
        ]
        assert "observed_training_representations" not in body
        return validate(
            {
                "categorical_value_map": [
                    {"raw_value": "new label", "canonical_value": "lower"}
                ]
            }
        )

    harmonized, definitions, report = stage2_analysis._harmonize_training_extraction(
        extracted=extracted,
        definitions=[_generic_feature_with_harmonization_plan()],
        output_dir=tmp_path / "harmonization",
        request_json=request_json,
        max_prompt_chars=20_000,
    )

    assert jobs == ["extend_stage2_harmonization_map_for_new_text_values"]
    assert report["plans_requested_from_llm"] == 1
    assert report["harmonization_validation_fallbacks"] == 0
    assert harmonized["assay_measurement"].tolist() == [
        "lower",
        "upper",
        "lower",
        "upper",
        "lower",
    ]
    mapping = {
        row["raw_value"]: row["canonical_value"]
        for row in definitions[0]["harmonization_plan"]["categorical_value_map"]
    }
    assert mapping == {
        "lower label": "lower",
        "new label": "lower",
        "upper label": "upper",
    }


def test_harmonization_retains_prior_plan_and_nulls_failed_delta(tmp_path: Path):
    extracted = pd.DataFrame(
        {
            "_oci_row_id": list(range(5)),
            "assay_measurement": [10.0, 75.0, "lower label", "upper label", "new label"],
        }
    )

    def invalid_delta(*_args, **_kwargs):
        raise stage2_analysis.Stage2ResponseValidationError(
            "mapping response remained invalid after bounded repairs"
        )

    harmonized, definitions, report = stage2_analysis._harmonize_training_extraction(
        extracted=extracted,
        definitions=[_generic_feature_with_harmonization_plan()],
        output_dir=tmp_path / "harmonization",
        request_json=invalid_delta,
        max_prompt_chars=20_000,
    )

    fallback = definitions[0]["harmonization_fallback"]
    assert fallback["status"] == "prior_plan_extended_with_null_mappings"
    assert fallback["unresolved_raw_values"] == ["new label"]
    assert report["harmonization_validation_fallbacks"] == 1
    assert report["features_with_harmonization_validation_fallback"] == ["feature_assay"]
    assert harmonized["assay_measurement"].tolist()[:4] == [
        "lower",
        "upper",
        "lower",
        "upper",
    ]
    assert pd.isna(harmonized["assay_measurement"].iloc[4])
    persisted = json.loads(
        (tmp_path / "harmonization" / "feature_assay" / "fallback.json").read_text()
    )
    assert persisted == fallback
    completion = json.loads(
        (tmp_path / "harmonization" / "feature_assay" / "complete.json").read_text()
    )
    assert completion["status"] == "complete_with_validation_fallback"


def test_harmonization_uses_audited_hybrid_fallback_without_prior_plan(
    tmp_path: Path,
):
    definition = {
        "feature_id": "feature_assay",
        "name": "assay_measurement",
        "description": "A pretreatment assay measurement.",
        "value_type": "continuous",
        "categories_or_unit": ["units"],
        "modeling_strategy": "continuous",
        "roles": ["prognostic"],
        "measurement_definition": "Extract the documented pretreatment assay measurement.",
        "missing_value_rule": "Return null when undocumented.",
    }
    extracted = pd.DataFrame(
        {
            "_oci_row_id": [0, 1, 2],
            "assay_measurement": [10.0, "lower label", "upper label"],
        }
    )

    def invalid_plan(*_args, **_kwargs):
        raise stage2_analysis.Stage2ResponseValidationError(
            "plan response remained invalid after bounded repairs"
        )

    harmonized, definitions, report = stage2_analysis._harmonize_training_extraction(
        extracted=extracted,
        definitions=[definition],
        output_dir=tmp_path / "round_001" / "harmonization",
        request_json=invalid_plan,
        max_prompt_chars=20_000,
    )

    assert definitions[0]["modeling_strategy"] == "continuous_with_categorical_fallback"
    assert "harmonization_plan" not in definitions[0]
    fallback = definitions[0]["harmonization_fallback"]
    assert fallback["status"] == "hybrid_modeling_without_harmonization_plan"
    assert fallback["unresolved_value_rule"] == "retain_raw_hybrid_value"
    assert report["harmonization_validation_fallbacks"] == 1
    assert harmonized["assay_measurement"].tolist() == [
        10.0,
        "lower label",
        "upper label",
    ]

    cached, cached_definitions, cached_report = (
        stage2_analysis._harmonize_training_extraction(
            extracted=extracted,
            definitions=[definition],
            output_dir=tmp_path / "round_001" / "harmonization",
            request_json=lambda *_args, **_kwargs: pytest.fail(
                "the completed fallback checkpoint should be reused"
            ),
            max_prompt_chars=20_000,
        )
    )
    assert cached_definitions[0]["modeling_strategy"] == (
        "continuous_with_categorical_fallback"
    )
    assert cached_report["plans_requested_from_llm"] == 0
    assert cached["assay_measurement"].tolist() == harmonized["assay_measurement"].tolist()

    reused, reused_definitions, reused_report = (
        stage2_analysis._harmonize_training_extraction(
            extracted=extracted,
            definitions=definitions,
            output_dir=tmp_path / "round_002" / "harmonization",
            request_json=lambda *_args, **_kwargs: pytest.fail(
                "a recorded hybrid fallback should prevent repeated plan requests"
            ),
            max_prompt_chars=20_000,
        )
    )
    assert reused_definitions == definitions
    assert reused_report["plans_requested_from_llm"] == 0
    assert reused_report["fallbacks"][0]["reused_from_prior_round"] is True
    assert reused["assay_measurement"].tolist() == harmonized["assay_measurement"].tolist()


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


def test_stage2_posthoc_oracle_ite_reports_unavailable_for_real_dataset(tmp_path: Path):
    prediction_path = tmp_path / "cross_fitted_predictions.csv"
    pd.DataFrame(
        {
            "_oci_row_id": [0, 1],
            "outer_fold": [1, 1],
            "estimated_cate": [0.1, 0.2],
        }
    ).to_csv(prediction_path, index=False)

    evaluation = stage2_workflow._evaluate_stage2_oracle_ite(
        prediction_path=prediction_path,
        dataset=pd.DataFrame({"outcome": [0, 1]}),
        output_dir=tmp_path,
    )

    assert evaluation["available"] is False
    assert "does not contain" in evaluation["reason"]
    assert (tmp_path / "posthoc_oracle_ite_metrics.json").is_file()
    assert not (tmp_path / "posthoc_predictions_with_oracle_ite.csv").exists()


def test_plain_stage2_is_fold_scoped_and_resumable(tmp_path: Path):
    handoff = tmp_path / "handoff.jsonl"
    rows = [
        {
            "source": "tfidf",
            "outer_fold": 1,
            "inner_fold": None,
            "scope": "full_outer_train",
            "evidence": {
                "architecture": "tfidf_topic_contrast",
                "evidence_id": "treatment-ecog",
                "objective": "treatment",
                "terms": ["ECOG", "poor performance status"],
            },
        },
        {
            "source": "text_models",
            "outer_fold": 1,
            "inner_fold": 1,
            "scope": "candidate_consistency_inner_train",
            "evidence": {
                "architecture": "embedding_contrast_whole",
                "evidence_id": "outcome-ecog",
                "objective": "outcome",
                "witnesses": ["ECOG 2 and unable to work"],
            },
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

    rows[0]["evidence"]["terms"].append("newly compiled evidence")
    handoff.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(RuntimeError, match="different evidence plan"):
        run_plain_handoff_stage2(
            handoff_path=handoff,
            output_dir=output,
            clinical_question="Identify confounders.",
            config=config,
            completion=_fake_completion(calls),
        )


def test_feature_definitions_resume_without_llm_calls_after_extractor_transport_change(
    tmp_path: Path,
):
    handoff = tmp_path / "handoff.jsonl"
    handoff.write_text(
        json.dumps(
            {
                "source": "tfidf",
                "outer_fold": 1,
                "inner_fold": None,
                "scope": "full_outer_train",
                "evidence": {
                    "architecture": "tfidf_topic_contrast",
                    "evidence_id": "treatment-ecog",
                    "objective": "treatment",
                    "terms": ["ECOG", "poor performance status"],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "stage2"
    calls = []

    def config(extraction_model, extraction_max_tokens):
        return PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="same-primary",
            extraction_max_tokens=extraction_max_tokens,
            max_prompt_chars=8_000,
            workers=2,
            required_architectures=(),
            extraction_llm=Stage2ExtractionLLMConfig(
                endpoint="http://extract.test/v1",
                model=extraction_model,
                workers=1,
            ),
        )

    run_plain_handoff_stage2(
        handoff_path=handoff,
        output_dir=output,
        clinical_question="Identify confounders.",
        config=config("extractor-a", 100_000),
        completion=_fake_completion(calls),
    )
    first_call_count = len(calls)
    first_completion = json.loads(
        (output / "outer_001" / "definitions_complete.json").read_text(
            encoding="utf-8"
        )
    )

    run_plain_handoff_stage2(
        handoff_path=handoff,
        output_dir=output,
        clinical_question="Identify confounders.",
        config=config("extractor-b", 60_000),
        completion=_fake_completion(calls),
    )

    second_completion = json.loads(
        (output / "outer_001" / "definitions_complete.json").read_text(
            encoding="utf-8"
        )
    )
    identity = json.loads((output / "model_identity.json").read_text(encoding="utf-8"))
    resumed_config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert len(calls) == first_call_count
    assert (
        second_completion["evidence_input_fingerprint"]
        == first_completion["evidence_input_fingerprint"]
    )
    assert identity["extraction"]["selected_model"] == "extractor-b"
    assert resumed_config["extraction_max_tokens"] == 60_000


def test_plain_stage2_runs_outer_folds_concurrently_and_writes_them_in_order(
    tmp_path: Path,
    monkeypatch,
):
    runner = PlainHandoffStage2(
        config=PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
            workers=3,
            required_architectures=(),
        ),
        clinical_question="Identify confounders.",
        completion=lambda _messages, _config: "{}",
    )
    packets = [{"outer_fold": outer_fold} for outer_fold in (3, 1, 2)]
    barrier = threading.Barrier(3)
    third_completed = threading.Event()
    second_completed = threading.Event()
    completion_order = []

    monkeypatch.setattr(
        runner,
        "_load_or_compile_evidence",
        lambda **_kwargs: (packets, {"status": "mocked"}),
    )

    def run_outer_fold(**kwargs):
        outer_fold = int(kwargs["outer_fold"])
        assert "agentic_evidence_workers" not in kwargs
        assert "agentic_evidence_executor" not in kwargs
        barrier.wait(timeout=2.0)
        if outer_fold == 2:
            assert third_completed.wait(timeout=2.0)
        elif outer_fold == 1:
            assert second_completed.wait(timeout=2.0)
        completion_order.append(outer_fold)
        if outer_fold == 3:
            third_completed.set()
        elif outer_fold == 2:
            second_completed.set()
        converged = outer_fold != 2
        return {
            "outer_fold": outer_fold,
            "features": [],
            "review_converged": converged,
            "review_convergence": {
                "status": "converged" if converged else "non_converged",
                "converged": converged,
            },
            "harmonization_validation_fallbacks": (
                [
                    {
                        "feature_id": "feature_generic_assay",
                        "status": "prior_plan_retained_with_unresolved_new_values",
                    }
                ]
                if outer_fold == 3
                else []
            ),
        }

    monkeypatch.setattr(runner, "_run_outer_fold", run_outer_fold)
    output_dir = tmp_path / "stage2"

    result = runner.run(
        handoff_path=tmp_path / "handoff.jsonl",
        output_dir=output_dir,
    )

    persisted = [
        json.loads(line)
        for line in (output_dir / "features_by_outer_fold.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert completion_order == [3, 2, 1]
    assert [row["outer_fold"] for row in persisted] == [1, 2, 3]
    assert result["outer_folds"] == 3
    assert result["nonconverged_outer_folds"] == [2]
    assert result["review_convergence_by_fold"]["2"]["converged"] is False
    assert result["outer_folds_with_harmonization_validation_fallbacks"] == [3]
    assert result["harmonization_validation_fallbacks_by_fold"]["3"] == [
        {
            "feature_id": "feature_generic_assay",
            "status": "prior_plan_retained_with_unresolved_new_values",
        }
    ]


def test_plain_stage2_finishes_extraction_review_and_causal_estimation(
    tmp_path: Path,
    monkeypatch,
):
    handoff = tmp_path / "handoff.jsonl"
    evidence_rows = []
    for outer_fold in (1, 2):
        for objective in ("treatment", "outcome"):
            evidence_rows.append(
                {
                    "source": "text_models",
                    "outer_fold": outer_fold,
                    "inner_fold": None,
                    "scope": "full_outer_train",
                    "evidence": {
                        "architecture": f"model_{objective}",
                        "objective": objective,
                        "witnesses": ["ECOG performance status"],
                    },
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
    primary_calls = []
    extraction_calls = []
    output = tmp_path / "stage2"
    config = PlainHandoffStage2Config(
        endpoint="http://stage2.test/v1",
        model="test-model",
        max_prompt_chars=12_000,
        workers=4,
        extraction_llm=stage2_workflow.Stage2ExtractionLLMConfig(
            endpoint="http://extract.test/v1",
            model="small-model",
            workers=4,
        ),
        max_review_rounds=1,
        estimation_trees=10,
        required_architectures=(),
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
    assert selection["confounder_univariable_screen"]["hard_selection_gate"] is False
    assert selection["multivariable_modifier_elastic_net_screen"][
        "hard_selection_gate"
    ] is False
    assert selection["final_role_assignment"] == "llm_all_evidence_adjudication"
    assert selection["role_adjudication"]["prompt_data_contract"][
        "oracle_columns_are_excluded"
    ] is True
    assert (
        output
        / "outer_001"
        / "selection"
        / "statistical_evidence.json"
    ).is_file()
    role_batch_prompt_path = output / "outer_001" / "selection" / "role_adjudication" / "batches" / "batch_001" / "prompt.json"
    assert role_batch_prompt_path.is_file()
    role_prompt = role_batch_prompt_path.read_text(encoding="utf-8").casefold()
    assert "oracle_ite" not in role_prompt
    assert "generation_config" not in role_prompt
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
    assert "analyze_stage2_variable_cluster" not in primary_calls
    assert "adjudicate_stage2_variable_role" not in primary_calls
    assert "adjudicate_stage2_roles_from_all_evidence" in primary_calls
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


def test_aggregate_supervisor_reuses_identical_feature_review_across_rounds(
    tmp_path: Path,
):
    definition = {
        "feature_id": "outer_001_feature_001",
        "name": "performance_status",
        "description": "Pretreatment performance status.",
        "value_type": "ordinal",
        "categories_or_unit": ["0", "1", "2", "3", "4"],
        "measurement_definition": "Extract the documented pretreatment ECOG score.",
        "missing_value_rule": "Return null when it is undocumented.",
    }
    summary = {
        "feature_id": definition["feature_id"],
        "name": definition["name"],
        "rows": 20,
        "nonmissing": 18,
        "nonmissing_fraction": 0.9,
        "unique_nonmissing": 3,
        "dominant_value_fraction": 0.5,
        "most_common_values": {"1": 9, "0": 6, "2": 3},
    }
    calls = []

    def request_json(messages, validate, *, request_kind="interpretation"):
        calls.append(prompt_inputs(messages)["feature"]["feature_id"])
        assert request_kind == "interpretation"
        return validate(
            {
                "action": "keep",
                "reason": "The aggregate extraction matches the declared ontology.",
            }
        )

    supervision = tmp_path / "ontology_supervision"
    first_definitions, first_changed, first_report = (
        stage2_analysis._request_aggregate_ontology_supervisor(
            definitions=[definition],
            summaries=[summary],
            failure_summary={"feature_failure_patterns": []},
            output_dir=supervision / "round_001" / "supervisor",
            request_json=request_json,
            workers=1,
            request_identity={"model": "primary-test-model"},
        )
    )
    second_definitions, second_changed, second_report = (
        stage2_analysis._request_aggregate_ontology_supervisor(
            definitions=[definition],
            summaries=[summary],
            failure_summary={"feature_failure_patterns": []},
            output_dir=supervision / "round_002" / "supervisor",
            request_json=request_json,
            workers=1,
            request_identity={"model": "primary-test-model"},
        )
    )
    changed_definition = {
        **definition,
        "measurement_definition": (
            "Extract the last documented pretreatment ECOG score before treatment."
        ),
    }
    _third_definitions, _third_changed, third_report = (
        stage2_analysis._request_aggregate_ontology_supervisor(
            definitions=[changed_definition],
            summaries=[summary],
            failure_summary={"feature_failure_patterns": []},
            output_dir=supervision / "round_003" / "supervisor",
            request_json=request_json,
            workers=1,
            request_identity={"model": "primary-test-model"},
        )
    )

    assert calls == [definition["feature_id"], definition["feature_id"]]
    assert first_definitions == second_definitions
    assert first_changed is second_changed is False
    assert first_report["model_requested_features"] == 1
    assert first_report["cache_reused_features"] == 0
    assert second_report["model_requested_features"] == 0
    assert second_report["cache_reused_features"] == 1
    assert third_report["model_requested_features"] == 1
    assert third_report["cache_reused_features"] == 0
    second_completion = json.loads(
        (
            supervision
            / "round_002"
            / "supervisor"
            / "feature_0001"
            / "complete.json"
        ).read_text(encoding="utf-8")
    )
    assert second_completion["reused"] is True
    assert second_completion["reuse_source"] == "content_addressed_cache"


def test_aggregate_supervisor_transport_exhaustion_is_not_checkpointed(
    tmp_path: Path,
):
    definition = {
        "feature_id": "outer_001_feature_001",
        "name": "performance_status",
        "description": "Pretreatment performance status.",
        "value_type": "ordinal",
        "categories_or_unit": ["0", "1", "2", "3", "4"],
        "measurement_definition": "Extract the documented pretreatment ECOG score.",
        "missing_value_rule": "Return null when it is undocumented.",
    }

    def exhausted(*_args, **_kwargs):
        raise stage2_analysis.Stage2RequestExhaustedError("extractor endpoint unavailable")

    output_dir = tmp_path / "ontology_supervision" / "round_001" / "supervisor"
    with pytest.raises(
        stage2_analysis.Stage2RequestExhaustedError,
        match="endpoint unavailable",
    ):
        stage2_analysis._request_aggregate_ontology_supervisor(
            definitions=[definition],
            summaries=[
                {
                    "feature_id": definition["feature_id"],
                    "name": definition["name"],
                }
            ],
            failure_summary={"feature_failure_patterns": []},
            output_dir=output_dir,
            request_json=exhausted,
            workers=1,
        )

    assert not (output_dir / "feature_0001" / "result.json").exists()
    assert not (output_dir / "feature_0001" / "complete.json").exists()
    assert not (output_dir / "complete.json").exists()


def test_aggregate_supervisor_can_revise_then_reextract_a_definition(
    tmp_path: Path,
    monkeypatch,
):
    rows = 48
    dataset = pd.DataFrame(
        {
            "patient_id": [f"p{index:02d}" for index in range(rows)],
            "clinical_text": [f"Pretreatment ECOG {index % 3}." for index in range(rows)],
            "treatment_indicator": [int(index % 3 != 0) for index in range(rows)],
            "outcome_indicator": [int(index % 3 != 0) for index in range(rows)],
        }
    )
    definitions = [
        {
            "feature_id": "outer_001_feature_001",
            "name": "performance_status",
            "description": "Baseline ECOG performance status.",
            "value_type": "ordinal",
            "categories_or_unit": ["ECOG 0", "ECOG 1", "ECOG 2"],
            "roles": [],
            "measurement_definition": "Extract the pretreatment ECOG score.",
            "missing_value_rule": "Use null when undocumented.",
            "supporting_packet_ids": ["packet_1"],
            "supporting_architectures": ["test"],
            "stability_summary": "training evidence",
            "caveats": "none",
        }
    ]
    split = {
        "outer_fold": 1,
        "fit_row_ids": list(range(24)),
        "heldout_row_ids": list(range(24, 48)),
        "inner_splits": [
            {
                "inner_fold": 1,
                "fit_row_ids": list(range(12)),
                "heldout_row_ids": list(range(12, 24)),
            },
            {
                "inner_fold": 2,
                "fit_row_ids": list(range(12, 24)),
                "heldout_row_ids": list(range(12)),
            },
        ],
    }
    jobs = []
    extraction_prompt_limits = []
    original_extract_rows = stage2_analysis.extract_rows

    def tracked_extract_rows(**kwargs):
        extraction_prompt_limits.append(kwargs["max_prompt_chars"])
        return original_extract_rows(**kwargs)

    monkeypatch.setattr(stage2_analysis, "extract_rows", tracked_extract_rows)

    def retain_revised_definition(**kwargs):
        selected = [
            {
                **kwargs["definitions"][0],
                "roles": ["confounder"],
                "nuisance_model_roles": ["treatment", "outcome"],
            }
        ]
        return (
            selected,
            {
                "schema_version": stage2_analysis.STAGE2_ROLE_SELECTION_SCHEMA_VERSION,
                "status": "complete",
            },
            selected,
            [],
        )

    monkeypatch.setattr(
        stage2_analysis,
        "select_stage2_features_elastic_net",
        retain_revised_definition,
    )

    def request_json(messages, validate, *, request_kind="interpretation"):
        body = prompt_inputs(messages)
        if body.get("task") in {
            "analyze_cluster",
            "outer_fold_role_adjudication",
            "adjudicate_stage2_roles_from_all_evidence",
        }:
            return validate(_agentic_fixture_response(body))
        jobs.append(body["job"])
        if body["job"] == "extract_stage2_patient_variables":
            assert request_kind == "extraction"
            response = {
                "rows": [
                    {
                        "row_id": patient["row_id"],
                        "values": {
                            "performance_status": "ECOG "
                            + re.search(r"ECOG\s*([0-2])", patient["text"]).group(1)
                        },
                    }
                    for patient in body["patients"]
                ]
            }
        elif (
            body["job"] == "review_stage2_small_model_extraction_ontology"
            and jobs.count("review_stage2_small_model_extraction_ontology") == 1
        ):
            assert request_kind == "interpretation"
            assert "patients" not in body
            assert "treatment_indicator" not in json.dumps(body).lower()
            response = {
                "action": "revise",
                "reason": "Clarify which pretreatment score to extract.",
                "description": "Last documented pretreatment ECOG performance status.",
                "value_type": "ordinal",
                "categories_or_unit": ["ECOG 0", "ECOG 1", "ECOG 2"],
                "measurement_definition": (
                    "Extract the last explicitly documented pretreatment ECOG score."
                ),
                "missing_value_rule": "Use null when no explicit score is documented.",
            }
        else:
            assert body["job"] == "review_stage2_small_model_extraction_ontology"
            assert request_kind == "interpretation"
            response = {
                "action": "keep",
                "reason": "The revised aggregate extraction is internally consistent.",
            }
        return validate(response)

    result = run_fold_analysis(
        dataset=dataset,
        definitions=definitions,
        split=split,
        clinical_question="Estimate treatment effect.",
        unit_id_column="patient_id",
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        outcome_type="binary",
        inner_folds=2,
        seed=11,
        output_dir=tmp_path / "outer_001",
        request_json=request_json,
        config=PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
            extraction_llm=Stage2ExtractionLLMConfig(
                endpoint="http://small-stage2.test/v1",
                model="small-test-model",
                workers=1,
            ),
            max_prompt_chars=12_000,
            extraction_max_prompt_chars=20_000,
            max_review_rounds=2,
            estimation_trees=10,
        ),
    )

    assert result["review_rounds"] == 2
    assert result["features"][0]["measurement_definition"].startswith("Extract the last")
    assert set(extraction_prompt_limits) == {20_000}
    assert jobs.count("review_stage2_small_model_extraction_ontology") == 2
    assert jobs.count("extract_stage2_patient_variables") > 48
    assert (
        tmp_path
        / "outer_001"
        / "ontology_supervision"
        / "round_002"
        / "supervisor"
        / "complete.json"
    ).is_file()
    round_two_extraction = json.loads(
        (
            tmp_path
            / "outer_001"
            / "ontology_supervision"
            / "round_002"
            / "extraction"
            / "complete.json"
        ).read_text(encoding="utf-8")
    )
    assert round_two_extraction["reextraction_scope"] == "changed_features_only"
    assert round_two_extraction["reextracted_feature_names"] == [
        "performance_status"
    ]


@pytest.mark.parametrize("multi_model", [False, True])
def test_fold_reselection_uses_frozen_preselection_without_training_extraction(
    tmp_path: Path,
    monkeypatch,
    multi_model,
):
    from dataclasses import replace

    from oci.inference.stage2_elastic_net_selection import Stage2ElasticNetSelectionConfig

    dataset = pd.DataFrame(
        {
            "patient_id": ["p0", "p1", "p2", "p3"],
            "clinical_text": ["baseline yes", "baseline no", "baseline yes", "heldout"],
            "treatment_indicator": [0, 1, 0, 1],
            "outcome_indicator": [0, 1, 1, 0],
        }
    )
    definitions = [
        {
            "feature_id": "outer_001_feature_001",
            "name": "baseline_status",
            "description": "Pretreatment baseline status.",
            "value_type": "binary",
            "categories_or_unit": ["Present", "Absent"],
            "roles": [],
            "measurement_definition": "Extract pretreatment baseline status.",
            "missing_value_rule": "Return null when undocumented.",
        }
    ]
    split = {
        "outer_fold": 1,
        "fit_row_ids": [0, 1, 2],
        "heldout_row_ids": [3],
        "inner_splits": [
            {"inner_fold": 1, "fit_row_ids": [0, 1], "heldout_row_ids": [2]},
            {"inner_fold": 2, "fit_row_ids": [1, 2], "heldout_row_ids": [0]},
        ],
    }
    packets = [{"packet_id": "packet_1", "outer_fold": 1}]
    config = PlainHandoffStage2Config(
        endpoint="http://stage2.test/v1",
        model="primary-model",
        extraction_llm=Stage2ExtractionLLMConfig(
            endpoint="http://extract.test/v1",
            model="extractor-model",
            workers=1,
        ),
        role_adjudication=Stage2RoleAdjudicationConfig(enabled=multi_model),
        statistical_selection=Stage2ElasticNetSelectionConfig(
            selection_mode="multi_model" if multi_model else "llm_roles"
        ),
        max_review_rounds=2,
        estimation_trees=10,
    )
    output_dir = tmp_path / "outer_001"
    matrix_path = output_dir / "extraction" / "all_candidates_fit" / "extracted.csv"
    matrix_path.parent.mkdir(parents=True)
    matrix = pd.DataFrame(
        {
            "_oci_row_id": [0, 1, 2],
            "baseline_status": ["Present", "Absent", "Present"],
        }
    )
    matrix.to_csv(matrix_path, index=False)
    convergence = {
        "schema_version": stage2_analysis.REVIEW_CONVERGENCE_SCHEMA_VERSION,
        "status": "converged",
        "converged": True,
        "review_rounds": 1,
        "maximum_review_rounds": 2,
        "continued_with_latest_ontology": False,
        "history": [{"round": 1}],
    }
    snapshot_value = {
        "schema_version": stage2_analysis.PRESELECTION_SNAPSHOT_SCHEMA_VERSION,
        "created_at": "2026-01-01T00:00:00Z",
        "outer_fold": 1,
        "source_selection_schema_version": "legacy-test",
        "source_selection_input_fingerprint": "legacy-selection",
        "source_reported_extracted_fit_fingerprint": "legacy-frame",
        "source_feature_definition_input_fingerprint": "feature-input",
        "source_feature_definitions_fingerprint": stage2_analysis._value_fingerprint(
            definitions
        ),
        "matrix_path": "extraction/all_candidates_fit/extracted.csv",
        "matrix_snapshot_fingerprint": stage2_analysis._frame_fingerprint(matrix),
        "fit_row_ids_fingerprint": stage2_analysis._value_fingerprint([0, 1, 2]),
        "fit_source_text_fingerprint": stage2_analysis._frame_fingerprint(
            dataset.iloc[[0, 1, 2]][["patient_id", "clinical_text"]].reset_index(
                drop=True
            )
        ),
        "treatment_outcome_fingerprint": stage2_analysis._frame_fingerprint(
            dataset.iloc[[0, 1, 2]][
                ["treatment_indicator", "outcome_indicator"]
            ].reset_index(drop=True)
        ),
        "inner_splits_fingerprint": stage2_analysis._value_fingerprint(
            split["inner_splits"]
        ),
        "stage1_packets_fingerprint": stage2_analysis._value_fingerprint(packets),
        "review_policy": stage2_analysis.frozen_preselection_review_policy(config),
        "review_policy_fingerprint": stage2_analysis._value_fingerprint(
            stage2_analysis.frozen_preselection_review_policy(config)
        ),
        "outcome_type": "binary",
        "rows": 3,
        "features": 1,
        "definitions": definitions,
        "review_metadata": {
            "review_rounds": 1,
            "evaluation_rounds": 1,
            "review_converged": True,
            "review_convergence": convergence,
            "ontology_refinement_rounds": 0,
            "harmonization_validation_fallbacks": [],
        },
    }
    snapshot = {
        **snapshot_value,
        "input_fingerprint": stage2_analysis._value_fingerprint(snapshot_value),
    }
    preselection_dir = output_dir / "preselection"
    preselection_dir.mkdir()
    (preselection_dir / "input.json").write_text(
        json.dumps(snapshot), encoding="utf-8"
    )
    (preselection_dir / "complete.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "schema_version": stage2_analysis.PRESELECTION_SNAPSHOT_SCHEMA_VERSION,
                "input_fingerprint": snapshot["input_fingerprint"],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        stage2_analysis,
        "_extract_training_with_ontology_feedback",
        lambda **_kwargs: pytest.fail("training extraction must not run"),
    )
    monkeypatch.setattr(
        stage2_analysis,
        "_request_aggregate_ontology_supervisor",
        lambda **_kwargs: pytest.fail("ontology supervision must not run"),
    )
    monkeypatch.setattr(
        stage2_analysis,
        "select_stage2_features_elastic_net",
        lambda **_kwargs: (
            [],
            {
                "schema_version": stage2_analysis.STAGE2_ROLE_SELECTION_SCHEMA_VERSION,
                "status": "complete",
            },
            [],
            [],
        ),
    )
    monkeypatch.setattr(
        stage2_analysis,
        "extract_rows",
        lambda **kwargs: pd.DataFrame({"_oci_row_id": kwargs["row_ids"]}),
    )
    monkeypatch.setattr(
        stage2_analysis,
        "_apply_harmonization_plans",
        lambda extracted, _definitions, **_kwargs: (extracted, {"plans": []}),
    )
    monkeypatch.setattr(
        stage2_analysis,
        "estimate_outer_fold",
        lambda **kwargs: {"status": "estimated_from_snapshot", "estimator": kwargs["estimator"]},
    )

    count_calls = []
    if multi_model:
        from oci.inference import stage2_modifier_count

        monkeypatch.setattr(stage2_analysis, "adjudicate_stage2_roles", lambda **kw: (
            [{**f, "roles": ["effect_modifier"]} for f in kw["definitions"]],
            {"decisions": [{"feature_id": f["feature_id"], "roles": ["effect_modifier"]}
                           for f in kw["definitions"]]}, {}))

        def choose_constant(**kw):
            assert kw["run_numerical"] is stage2_analysis._run_stage2_statistical_selection
            assert kw["extracted_fit"]._oci_row_id.tolist() == [0, 1, 2]
            assert kw["inner_splits"] == split["inner_splits"]
            assert kw["role_policy"].enabled
            assert kw["selected"][0]["roles"] == ["effect_modifier"]
            assert kw["estimation_trees"] in {10, 20}
            count_calls.append(kw["estimation_trees"])
            report = {"status": "complete", "chosen_modifier_count": 0, "chosen_estimator": "linear_interactions"}
            return [], {"decisions": [{"feature_id": definitions[0]["feature_id"], "roles": []}]}, report

        monkeypatch.setattr(stage2_modifier_count, "select_modifier_count", choose_constant)

    arguments = dict(
        dataset=dataset,
        definitions=definitions,
        split=split,
        clinical_question="Estimate treatment effect.",
        unit_id_column="patient_id",
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        outcome_type="binary",
        inner_folds=2,
        seed=11,
        output_dir=output_dir,
        request_json=lambda *_args, **_kwargs: pytest.fail(
            "the frozen no-feature fixture should make no LLM request"
        ),
        config=config,
        stage1_packets=packets,
    )
    result = run_fold_analysis(**arguments)

    assert result["estimation"]["status"] == "estimated_from_snapshot"
    assert result["estimation"]["estimator"] == ("linear_interactions" if multi_model else "causal_forest")
    assert result["review_convergence"]["reused_frozen_preselection_snapshot"] is True
    assert result["review_rounds"] == 1
    assert len(count_calls) == int(multi_model)
    if multi_model:
        assert result["selection"]["final_role_assignment"] == "llm_confounders_and_nested_r_loss_modifiers"
        assert result["selection"]["modifier_count_selection"]["chosen_modifier_count"] == 0
        resumed = run_fold_analysis(**arguments)
        assert resumed["estimation"]["estimator"] == "linear_interactions"
        assert count_calls == [10]
        run_fold_analysis(**{**arguments, "config": replace(config, estimation_trees=20)})
        assert count_calls == [10, 20]


def test_fold_reselection_reuses_archived_heldout_components_and_extracts_only_missing(
    tmp_path: Path,
    monkeypatch,
):
    dataset = pd.DataFrame(
        {
            "patient_id": ["p0", "p1", "p2", "p3"],
            "clinical_text": ["fit zero", "fit one", "fit two", "heldout"],
            "treatment_indicator": [0, 1, 0, 1],
            "outcome_indicator": [0, 1, 1, 0],
        }
    )
    definitions = [
        {
            "feature_id": "outer_001_feature_001",
            "name": "cached_component",
            "description": "A cached pretreatment component.",
            "value_type": "binary",
            "categories_or_unit": ["Present", "Absent"],
            "roles": [],
            "measurement_definition": "Extract the cached component.",
            "missing_value_rule": "Return null when undocumented.",
        },
        {
            "feature_id": "outer_001_feature_002",
            "name": "new_component",
            "description": "A newly required pretreatment component.",
            "value_type": "binary",
            "categories_or_unit": ["Present", "Absent"],
            "roles": [],
            "measurement_definition": "Extract the new component.",
            "missing_value_rule": "Return null when undocumented.",
        },
    ]
    split = {
        "outer_fold": 1,
        "fit_row_ids": [0, 1, 2],
        "heldout_row_ids": [3],
        "inner_splits": [
            {"inner_fold": 1, "fit_row_ids": [0, 1], "heldout_row_ids": [2]},
            {"inner_fold": 2, "fit_row_ids": [1, 2], "heldout_row_ids": [0]},
        ],
    }
    packets = [{"packet_id": "packet_1", "outer_fold": 1}]
    config = PlainHandoffStage2Config(
        endpoint="http://stage2.test/v1",
        model="primary-model",
        extraction_llm=Stage2ExtractionLLMConfig(
            endpoint="http://extract.test/v1",
            model="extractor-model",
            workers=1,
        ),
        role_adjudication=Stage2RoleAdjudicationConfig(enabled=False),
        max_review_rounds=2,
        estimation_trees=10,
    )
    stage2_dir = tmp_path / "stage2"
    output_dir = stage2_dir / "outer_001"
    matrix_path = output_dir / "extraction" / "all_candidates_fit" / "extracted.csv"
    matrix_path.parent.mkdir(parents=True)
    matrix = pd.DataFrame(
        {
            "_oci_row_id": [0, 1, 2],
            "cached_component": ["Present", "Absent", "Present"],
            "new_component": ["Absent", "Present", "Present"],
        }
    )
    matrix.to_csv(matrix_path, index=False)

    archive_relative = Path("reselection_archives") / "reselection_test"
    archive_dir = stage2_dir / archive_relative
    cached_raw = pd.DataFrame(
        {"_oci_row_id": [3], "cached_component": ["Absent"]}
    )
    cached_path = (
        archive_dir
        / "artifacts"
        / "outer_001"
        / "extraction"
        / "heldout"
        / "extracted.csv"
    )
    cached_path.parent.mkdir(parents=True)
    cached_raw.to_csv(cached_path, index=False)
    state = {
        "schema_version": "stage2_reselection_migration_v1",
        "status": "prepared",
        "reselection_id": "reselection_test",
        "archive_path": str(archive_relative),
        "policy_fingerprint": "test-policy",
        "outer_folds": [1],
    }
    (stage2_dir / "reselection_state.json").write_text(
        json.dumps(state), encoding="utf-8"
    )
    (archive_dir / "manifest.json").write_text(json.dumps(state), encoding="utf-8")
    heldout_cache = {
        "schema_version": stage2_analysis.HELDOUT_MEASUREMENT_CACHE_SCHEMA_VERSION,
        "source_artifact_path": "outer_001/extraction/heldout/extracted.csv",
        "raw_frame_fingerprint": stage2_analysis._frame_fingerprint(cached_raw),
        "heldout_row_ids": [3],
        "heldout_row_ids_fingerprint": stage2_analysis._value_fingerprint([3]),
        "heldout_source_text_fingerprint": stage2_analysis._frame_fingerprint(
            dataset.iloc[[3]][["patient_id", "clinical_text"]].reset_index(drop=True)
        ),
        "extraction_model": "extractor-model",
        "rows": 1,
        "features": 1,
        "measurement_definitions": [
            stage2_analysis.frozen_measurement_definition_identity(definitions[0])
        ],
    }
    convergence = {
        "schema_version": stage2_analysis.REVIEW_CONVERGENCE_SCHEMA_VERSION,
        "status": "converged",
        "converged": True,
        "review_rounds": 1,
        "maximum_review_rounds": 2,
        "continued_with_latest_ontology": False,
        "history": [{"round": 1}],
    }
    snapshot_value = {
        "schema_version": stage2_analysis.PRESELECTION_SNAPSHOT_SCHEMA_VERSION,
        "created_at": "2026-01-01T00:00:00Z",
        "outer_fold": 1,
        "source_selection_schema_version": "legacy-test",
        "source_selection_input_fingerprint": "legacy-selection",
        "source_reported_extracted_fit_fingerprint": "legacy-frame",
        "source_feature_definition_input_fingerprint": "feature-input",
        "source_feature_definitions_fingerprint": stage2_analysis._value_fingerprint(
            definitions
        ),
        "matrix_path": "extraction/all_candidates_fit/extracted.csv",
        "matrix_snapshot_fingerprint": stage2_analysis._frame_fingerprint(matrix),
        "fit_row_ids_fingerprint": stage2_analysis._value_fingerprint([0, 1, 2]),
        "fit_source_text_fingerprint": stage2_analysis._frame_fingerprint(
            dataset.iloc[[0, 1, 2]][["patient_id", "clinical_text"]].reset_index(
                drop=True
            )
        ),
        "treatment_outcome_fingerprint": stage2_analysis._frame_fingerprint(
            dataset.iloc[[0, 1, 2]][
                ["treatment_indicator", "outcome_indicator"]
            ].reset_index(drop=True)
        ),
        "inner_splits_fingerprint": stage2_analysis._value_fingerprint(
            split["inner_splits"]
        ),
        "stage1_packets_fingerprint": stage2_analysis._value_fingerprint(packets),
        "review_policy": stage2_analysis.frozen_preselection_review_policy(config),
        "review_policy_fingerprint": stage2_analysis._value_fingerprint(
            stage2_analysis.frozen_preselection_review_policy(config)
        ),
        "outcome_type": "binary",
        "rows": 3,
        "features": 2,
        "definitions": definitions,
        "review_metadata": {
            "review_rounds": 1,
            "evaluation_rounds": 1,
            "review_converged": True,
            "review_convergence": convergence,
            "ontology_refinement_rounds": 0,
            "harmonization_validation_fallbacks": [],
        },
        "heldout_measurement_cache": heldout_cache,
    }
    snapshot = {
        **snapshot_value,
        "input_fingerprint": stage2_analysis._value_fingerprint(snapshot_value),
    }
    preselection_dir = output_dir / "preselection"
    preselection_dir.mkdir()
    (preselection_dir / "input.json").write_text(
        json.dumps(snapshot), encoding="utf-8"
    )
    (preselection_dir / "complete.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "schema_version": stage2_analysis.PRESELECTION_SNAPSHOT_SCHEMA_VERSION,
                "input_fingerprint": snapshot["input_fingerprint"],
            }
        ),
        encoding="utf-8",
    )

    selected = [{**definition, "roles": ["confounder"]} for definition in definitions]
    monkeypatch.setattr(
        stage2_analysis,
        "select_stage2_features_elastic_net",
        lambda **_kwargs: (
            selected,
            {
                "schema_version": stage2_analysis.STAGE2_ROLE_SELECTION_SCHEMA_VERSION,
                "status": "complete",
            },
            definitions,
            [],
        ),
    )
    extraction_calls = []

    def extract_missing(**kwargs):
        names = [definition["name"] for definition in kwargs["definitions"]]
        extraction_calls.append(names)
        return pd.DataFrame(
            {
                "_oci_row_id": kwargs["row_ids"],
                **{name: ["Present"] for name in names},
            }
        )

    monkeypatch.setattr(stage2_analysis, "extract_rows", extract_missing)
    monkeypatch.setattr(
        stage2_analysis,
        "_apply_harmonization_plans",
        lambda extracted, _definitions, **_kwargs: (extracted, {"plans": []}),
    )
    captured_heldout = {}

    def estimate(**kwargs):
        captured_heldout["frame"] = kwargs["extracted_heldout"].copy()
        return {"status": "estimated_with_reuse"}

    monkeypatch.setattr(stage2_analysis, "estimate_outer_fold", estimate)

    result = run_fold_analysis(
        dataset=dataset,
        definitions=definitions,
        split=split,
        clinical_question="Estimate treatment effect.",
        unit_id_column="patient_id",
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        outcome_type="binary",
        inner_folds=2,
        seed=11,
        output_dir=output_dir,
        request_json=lambda *_args, **_kwargs: pytest.fail(
            "only the mocked missing-component extractor should run"
        ),
        config=config,
        stage1_packets=packets,
    )

    assert result["estimation"]["status"] == "estimated_with_reuse"
    assert extraction_calls == [["new_component"]]
    assert captured_heldout["frame"].iloc[0].to_dict() == {
        "_oci_row_id": 3,
        "cached_component": "Absent",
        "new_component": "Present",
    }
    reuse = json.loads(
        (
            output_dir / "extraction" / "heldout" / "measurement_reuse.json"
        ).read_text(encoding="utf-8")
    )
    assert reuse["reused_features"] == ["cached_component"]
    assert reuse["newly_extracted_features"] == ["new_component"]

    all_cached = stage2_analysis._extract_outer_heldout_measurements(
        dataset=dataset,
        heldout_ids=[3],
        text_column="clinical_text",
        measurement_definitions=[definitions[0]],
        output_dir=stage2_dir / "outer_all_cached",
        request_json=lambda *_args, **_kwargs: pytest.fail(
            "an entirely cached dependency set must make no LLM request"
        ),
        workers=1,
        max_prompt_chars=10_000,
        feature_batch_size=10,
        request_identity={"model": "extractor-model"},
        tokenizer=None,
        frozen_cache=heldout_cache,
        serial_extraction={},
    )
    assert extraction_calls == [["new_component"]]
    assert all_cached.iloc[0].to_dict() == {
        "_oci_row_id": 3,
        "cached_component": "Absent",
    }
    all_cached_audit = json.loads(
        (
            stage2_dir
            / "outer_all_cached"
            / "extraction"
            / "heldout"
            / "measurement_reuse.json"
        ).read_text(encoding="utf-8")
    )
    assert all_cached_audit["newly_extracted_feature_count"] == 0

    changed_definition = {
        **definitions[0],
        "measurement_definition": "Extract a materially changed component.",
    }
    changed = stage2_analysis._extract_outer_heldout_measurements(
        dataset=dataset,
        heldout_ids=[3],
        text_column="clinical_text",
        measurement_definitions=[changed_definition],
        output_dir=stage2_dir / "outer_changed_definition",
        request_json=lambda *_args, **_kwargs: pytest.fail(
            "the mocked extractor handles definition-incompatible cache misses"
        ),
        workers=1,
        max_prompt_chars=10_000,
        feature_batch_size=10,
        request_identity={"model": "extractor-model"},
        tokenizer=None,
        frozen_cache=heldout_cache,
        serial_extraction={},
    )
    assert extraction_calls == [["new_component"], ["cached_component"]]
    assert changed["cached_component"].tolist() == ["Present"]
    changed_audit = json.loads(
        (
            stage2_dir
            / "outer_changed_definition"
            / "extraction"
            / "heldout"
            / "measurement_reuse.json"
        ).read_text(encoding="utf-8")
    )
    assert changed_audit["definition_incompatible_features"] == [
        "cached_component"
    ]


def test_repeated_training_extraction_failures_refine_ontology_and_reextract(
    tmp_path: Path,
):
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

    def request_json(messages, validate, *, request_kind="interpretation"):
        body = prompt_inputs(messages)
        if body.get("task") in {
            "analyze_cluster",
            "outer_fold_role_adjudication",
            "adjudicate_stage2_roles_from_all_evidence",
        }:
            return validate(_agentic_fixture_response(body))
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


def test_failure_refinement_reextracts_only_changed_features_and_resumes(
    tmp_path: Path,
):
    dataset = pd.DataFrame(
        {
            "clinical_text": [
                f"Pretreatment stage {'III' if index % 2 == 0 else 'IV'}; marker present."
                for index in range(4)
            ]
        }
    )
    definitions = [
        {
            "feature_id": "outer_001_feature_001",
            "name": "disease_stage",
            "description": "Pretreatment overall disease stage.",
            "value_type": "ordinal",
            "categories_or_unit": ["stage_i", "stage_ii"],
            "measurement_definition": "Extract the documented pretreatment stage.",
            "missing_value_rule": "Use null when the stage is undocumented.",
        },
        {
            "feature_id": "outer_001_feature_002",
            "name": "stable_marker",
            "description": "Pretreatment marker status.",
            "value_type": "binary",
            "categories_or_unit": ["absent", "present"],
            "measurement_definition": "Extract the documented pretreatment marker.",
            "missing_value_rule": "Use null when the marker is undocumented.",
        },
    ]
    extraction_feature_sets = []

    def request_json(messages, validate, *, request_kind="interpretation"):
        body = prompt_inputs(messages)
        if body.get("task") in {
            "analyze_cluster",
            "outer_fold_role_adjudication",
            "adjudicate_stage2_roles_from_all_evidence",
        }:
            return validate(_agentic_fixture_response(body))
        if body["job"] == "extract_stage2_patient_variables":
            assert request_kind == "extraction"
            feature_names = [feature["name"] for feature in body["features"]]
            extraction_feature_sets.append(feature_names)
            categories_by_name = {
                feature["name"]: feature["categories_or_unit"]
                for feature in body["features"]
            }
            rows = []
            for patient in body["patients"]:
                values = {}
                if "disease_stage" in feature_names:
                    raw = "stage_iii" if "stage III" in patient["text"] else "stage_iv"
                    values["disease_stage"] = (
                        raw
                        if raw in categories_by_name["disease_stage"]
                        else raw.replace("stage_", "TNM stage ").upper()
                    )
                if "stable_marker" in feature_names:
                    values["stable_marker"] = "present"
                rows.append({"row_id": patient["row_id"], "values": values})
            return validate({"rows": rows})
        if body["job"] == "map_extracted_values_to_declared_category_ontology":
            return validate(
                {
                    "corrections": [
                        {"mapping_id": item["mapping_id"], "value": None}
                        for item in body["items"]
                    ]
                }
            )
        assert body["job"] == (
            "refine_stage2_feature_ontology_from_repeated_extraction_failures"
        )
        assert body["feature"]["name"] == "disease_stage"
        return validate(
            {
                "action": "revise",
                "reason": "Add the observed pretreatment stage groups.",
                "description": "Pretreatment overall disease stage group.",
                "value_type": "ordinal",
                "categories_or_unit": ["stage_iii", "stage_iv"],
                "measurement_definition": (
                    "Extract the documented pretreatment overall stage group."
                ),
                "missing_value_rule": "Use null when the stage group is undocumented.",
            }
        )

    output_dir = tmp_path / "initial"
    feedback_dir = tmp_path / "refinement"
    extracted, refined, rounds = stage2_analysis._extract_training_with_ontology_feedback(
        dataset=dataset,
        row_ids=list(range(4)),
        text_column="clinical_text",
        definitions=definitions,
        output_dir=output_dir,
        feedback_dir=feedback_dir,
        request_json=request_json,
        workers=2,
        max_prompt_chars=20_000,
        feature_batch_size=10,
        minimum_failure_patients=3,
        max_refinement_rounds=2,
        request_identity={"model": "small-test-model"},
        tokenizer=_CharacterChatTokenizer(),
    )

    assert rounds == 1
    assert [feature["name"] for feature in refined] == [
        "disease_stage",
        "stable_marker",
    ]
    assert extracted["disease_stage"].isin(["stage_iii", "stage_iv"]).all()
    assert extracted["stable_marker"].eq("present").all()
    assert extraction_feature_sets[:4] == [
        ["disease_stage", "stable_marker"]
    ] * 4
    assert extraction_feature_sets[4:] == [["disease_stage"]] * 4

    delta_root = feedback_dir / "round_001" / "extraction"
    completion = json.loads((delta_root / "complete.json").read_text(encoding="utf-8"))
    assert completion["reextraction_scope"] == "changed_features_only"
    assert completion["reextracted_feature_names"] == ["disease_stage"]
    assert completion["reused_features"] == 1
    assert (
        delta_root / "changed_features" / "batches" / "batch_00001" / "complete.json"
    ).is_file()
    merged = pd.read_csv(delta_root / "extracted.csv")
    assert list(merged.columns) == ["_oci_row_id", "disease_stage", "stable_marker"]
    final_summary = json.loads(
        (feedback_dir / "final_failure_summary.json").read_text(encoding="utf-8")
    )
    assert final_summary["feature_failure_patterns"] == []

    def unexpected_request(_messages, _validate, *, request_kind="interpretation"):
        raise AssertionError(f"completed delta checkpoint should resume ({request_kind})")

    resumed, resumed_definitions, resumed_rounds = (
        stage2_analysis._extract_training_with_ontology_feedback(
            dataset=dataset,
            row_ids=list(range(4)),
            text_column="clinical_text",
            definitions=definitions,
            output_dir=output_dir,
            feedback_dir=feedback_dir,
            request_json=unexpected_request,
            workers=2,
            max_prompt_chars=20_000,
            feature_batch_size=10,
            minimum_failure_patients=3,
            max_refinement_rounds=2,
            request_identity={"model": "small-test-model"},
            tokenizer=_CharacterChatTokenizer(),
        )
    )
    pd.testing.assert_frame_equal(resumed, extracted)
    assert resumed_definitions == refined
    assert resumed_rounds == 1


def test_incremental_failure_summary_replaces_only_changed_feature_patterns():
    prior = {
        "issue_files": 8,
        "feature_failure_patterns": [
            {
                "feature_name": "changed_feature",
                "failure_kind": "out_of_ontology_category",
                "reason": "old changed failure",
                "patient_count": 4,
                "patient_row_ids": [0, 1, 2, 3],
            },
            {
                "feature_name": "reused_feature",
                "failure_kind": "invalid_scalar_or_value_type",
                "reason": "retained reused failure",
                "patient_count": 3,
                "patient_row_ids": [1, 2, 3],
            },
            {
                "feature_name": "dropped_feature",
                "failure_kind": "out_of_ontology_category",
                "reason": "must be dropped",
                "patient_count": 5,
                "patient_row_ids": [0, 1, 2, 3, 4],
            },
        ],
        "structural_failure_patient_row_ids": [1, 4],
    }
    delta = {
        "issue_files": 2,
        "feature_failure_patterns": [
            {
                "feature_name": "changed_feature",
                "failure_kind": "out_of_ontology_category",
                "reason": "new changed failure",
                "patient_count": 1,
                "patient_row_ids": [5],
            }
        ],
        "structural_failure_patient_row_ids": [4, 5],
    }

    merged = stage2_analysis._merge_incremental_failure_summaries(
        prior_summary=prior,
        delta_summary=delta,
        current_feature_names=["changed_feature", "reused_feature"],
        changed_feature_names=["changed_feature"],
    )

    assert merged["issue_files"] == 10
    assert [row["reason"] for row in merged["feature_failure_patterns"]] == [
        "retained reused failure",
        "new changed failure",
    ]
    assert merged["structural_failure_patient_row_ids"] == [1, 4, 5]
    assert merged["incremental_refinement"]["reextracted_feature_names"] == [
        "changed_feature"
    ]
    assert merged["incremental_refinement"]["reused_feature_names"] == [
        "reused_feature"
    ]


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


def test_final_training_extraction_fails_fast_when_effectively_all_null(
    tmp_path: Path,
):
    dataset = pd.DataFrame(
        {
            "patient_id": [f"p{index:02d}" for index in range(12)],
            "clinical_text": ["No supported baseline feature."] * 12,
            "treatment_indicator": [index % 2 for index in range(12)],
            "outcome_indicator": [(index // 2) % 2 for index in range(12)],
        }
    )
    definition = {
        "feature_id": "outer_001_feature_001",
        "name": "age",
        "description": "Age at treatment.",
        "value_type": "continuous",
        "categories_or_unit": ["years"],
        "roles": ["confounder", "effect_modifier"],
        "measurement_definition": "Extract age in years.",
        "missing_value_rule": "Use null when undocumented.",
        "configured_explicit_feature": True,
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

    def request_json(messages, validate, *, request_kind="interpretation"):
        body = prompt_inputs(messages)
        if body.get("task") in {
            "analyze_cluster",
            "outer_fold_role_adjudication",
            "adjudicate_stage2_roles_from_all_evidence",
        }:
            return validate(_agentic_fixture_response(body))
        if body["job"] == "extract_stage2_patient_variables":
            return validate(
                {
                    "rows": [
                        {"row_id": patient["row_id"], "values": {"age": None}}
                        for patient in body["patients"]
                    ]
                }
            )
        return validate(
            {
                "feature_decisions": [
                    {
                        "feature_id": "outer_001_feature_001",
                        "action": "keep",
                        "reason": "Retain for the final health check.",
                        "modeling_strategy": "continuous",
                    }
                ],
                "overall_assessment": "No measured values.",
            }
        )

    output = tmp_path / "outer_001"
    with pytest.raises(ValueError, match="catastrophically sparse"):
        run_fold_analysis(
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
            seed=19,
            output_dir=output,
            request_json=request_json,
            config=PlainHandoffStage2Config(
                endpoint="http://stage2.test/v1",
                model="test-model",
                workers=1,
                max_review_rounds=1,
                estimation_trees=10,
            ),
        )

    health = json.loads((output / "extraction" / "fit_health.json").read_text(encoding="utf-8"))
    assert health["status"] == "failed"
    assert health["all_null_rows"] == 6
    assert not (output / "estimation" / "predictions.csv").exists()
