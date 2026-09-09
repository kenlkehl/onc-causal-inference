from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import oci.inference.plain_handoff_stage2 as stage2_workflow

from oci.inference.plain_handoff_stage2 import (
    PlainHandoffStage2,
    PlainHandoffStage2Config,
    Stage2ExtractionLLMConfig,
    run_plain_handoff_stage2,
)
from oci.inference.plain_handoff_stage2_analysis import (
    ESTIMATION_CHECKPOINT_SCHEMA_VERSION,
    Stage2RequestExhaustedError,
)
from oci.inference.stage2_sequential_consolidation import SELECTION_SCHEMA_VERSION
from oci.inference.vllm_server_pool import (
    ManagedVLLMConfig,
    validate_managed_vllm_pool_isolation,
)


def _runner(**config_overrides) -> PlainHandoffStage2:
    return PlainHandoffStage2(
        config=PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
            required_architectures=(),
            **config_overrides,
        ),
        clinical_question="Identify pretreatment confounders and effect modifiers.",
        completion=lambda _messages, _config: "{}",
    )


def test_completed_outer_fold_revalidates_selection_and_estimation_inputs(
    tmp_path: Path,
    monkeypatch,
):
    runner = _runner(propensity_clip=0.49, estimation_trees=999)
    output_dir = tmp_path / "outer_001"
    output_dir.mkdir(parents=True)
    definition = {
        "feature_id": "outer_001_feature_001",
        "name": "performance_status",
        "roles": ["confounder"],
    }
    definition_input = stage2_workflow._feature_definition_input_value(
        config=runner.config,
        clinical_question=runner.clinical_question,
        outer_fold=1,
        discovery_packets=[],
        seed=42,
    )
    evidence_fingerprint = stage2_workflow._value_fingerprint(definition_input)
    (output_dir / "feature_definitions.json").write_text(
        json.dumps({"features": [definition], "candidate_dispositions": {}}),
        encoding="utf-8",
    )
    (output_dir / "definitions_complete.json").write_text(
        json.dumps({"evidence_input_fingerprint": evidence_fingerprint}),
        encoding="utf-8",
    )
    (output_dir / "final_definitions.json").write_text(
        json.dumps({"features": [definition], "review_rounds": 1}),
        encoding="utf-8",
    )
    (output_dir / "complete.json").write_text(
        json.dumps({"status": "complete", "phase": "causal_estimation"}),
        encoding="utf-8",
    )

    selection_dir = output_dir / "selection"
    selection_dir.mkdir()
    stale_selection_input = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "temporal_scope": runner.config.input_temporal_scope,
        "extracted_fit_fingerprint": "stale-extraction",
        "treatment_outcome_fingerprint": "stale-outcomes",
        "stage1_packets_fingerprint": stage2_workflow._value_fingerprint([]),
        "definitions": [definition],
        "inner_splits": [{"fit_row_ids": [0], "heldout_row_ids": [1]}],
        "outcome_type": "binary",
        "selection_consolidation_policy": (
            runner.config.selection_consolidation.scientific_dict()
        ),
        "selection_consolidation_llm_model": runner.config.model,
        "statistical_selection_policy": (
            runner.config.statistical_selection.public_dict()
        ),
    }
    selection_fingerprint = stage2_workflow._value_fingerprint(
        stale_selection_input
    )
    (selection_dir / "input.json").write_text(
        json.dumps(
            {
                **stale_selection_input,
                "input_fingerprint": selection_fingerprint,
            }
        ),
        encoding="utf-8",
    )
    (selection_dir / "complete.json").write_text(
        json.dumps(
            {
                "schema_version": SELECTION_SCHEMA_VERSION,
                "input_fingerprint": selection_fingerprint,
            }
        ),
        encoding="utf-8",
    )

    estimation_dir = output_dir / "estimation"
    estimation_dir.mkdir()
    (estimation_dir / "complete.json").write_text(
        json.dumps(
            {
                "schema_version": ESTIMATION_CHECKPOINT_SCHEMA_VERSION,
                "outcome_type": "binary",
                "input_fingerprint": "stale-estimation",
            }
        ),
        encoding="utf-8",
    )
    (estimation_dir / "diagnostics.json").write_text(
        json.dumps(
            {
                "sentinel": "stale",
                "propensity_clip": 0.01,
                "estimation_trees": 10,
            }
        ),
        encoding="utf-8",
    )

    analysis_calls = []

    def revalidate_analysis(**kwargs):
        analysis_calls.append(kwargs)
        return {
            "features": [definition],
            "review_rounds": 1,
            "evaluation_rounds": 1,
            "review_converged": True,
            "review_convergence": {"converged": True},
            "ontology_refinement_rounds": 0,
            "harmonization_validation_fallbacks": [],
            "estimation": {
                "sentinel": "current",
                "propensity_clip": kwargs["config"].propensity_clip,
                "estimation_trees": kwargs["config"].estimation_trees,
            },
        }

    monkeypatch.setattr(stage2_workflow, "run_fold_analysis", revalidate_analysis)
    dataset = pd.DataFrame(
        {
            "patient_id": ["a", "b"],
            "clinical_text": ["old text", "new text"],
            "treatment_indicator": [0, 1],
            "outcome_indicator": [1, 0],
        }
    )
    result = runner._run_outer_fold(
        outer_fold=1,
        packets=[],
        output_dir=output_dir,
        dataset=dataset,
        split={
            "fit_row_ids": [0],
            "heldout_row_ids": [1],
            "inner_splits": [],
        },
    )

    assert len(analysis_calls) == 1
    assert analysis_calls[0]["dataset"] is dataset
    assert result["estimation"] == {
        "sentinel": "current",
        "propensity_clip": 0.49,
        "estimation_trees": 999,
    }


def test_evidence_compilation_invalidates_when_embedding_cache_appears(
    tmp_path: Path,
    monkeypatch,
):
    stage1_root = tmp_path / "stage1"
    handoff_path = stage1_root / "handoff" / "researcher_handoff.jsonl"
    handoff_path.parent.mkdir(parents=True)
    handoff_path.write_text("{}\n", encoding="utf-8")
    output_dir = tmp_path / "stage2"
    compile_calls = []
    packet = {"packet_id": "packet-1", "outer_fold": 1}

    def compile_evidence(*_args, **_kwargs):
        compile_calls.append("compile")
        return SimpleNamespace(
            packets=(packet,),
            cards_by_outer_fold={1: (packet,)},
            members_by_outer_fold={1: (packet,)},
            lineage_by_outer_fold={1: (packet,)},
            summary={},
        )

    monkeypatch.setattr(
        stage2_workflow,
        "compile_stage2_handoff_evidence",
        compile_evidence,
    )
    runner = _runner()
    runner._load_or_compile_evidence(
        handoff_path=handoff_path,
        output_dir=output_dir,
        seed=42,
    )
    runner._load_or_compile_evidence(
        handoff_path=handoff_path,
        output_dir=output_dir,
        seed=42,
    )
    assert compile_calls == ["compile"]

    cache_dir = (
        stage1_root
        / "components"
        / "embedding_cache"
        / "cache"
        / "semantic-cache"
    )
    cache_dir.mkdir(parents=True)
    (cache_dir / "metadata.json").write_text(
        json.dumps(
            {
                "sentence_model_name": "new-model",
                "hidden_size": 2,
                "num_samples": 2,
                "total_chunks": 2,
                "cache_hash": "semantic-input-v2",
                "chunking_policy_version": "v2",
                "dtype": "float16",
            }
        ),
        encoding="utf-8",
    )
    # The dependency identity only stats these products; the compilation-cache
    # check must not read or hash their contents.
    (cache_dir / "chunk_embeddings.npy").write_bytes(b"not-read")
    (cache_dir / "offsets.npy").write_bytes(b"not-read")

    _packets, summary = runner._load_or_compile_evidence(
        handoff_path=handoff_path,
        output_dir=output_dir,
        seed=42,
    )

    assert compile_calls == ["compile", "compile"]
    dependency = summary["compiler_signature"][
        "stage1_embedding_cache_dependency"
    ]
    assert dependency["semantic_metadata"]["sentence_model_name"] == "new-model"
    assert dependency["products"]["chunk_embeddings.npy"]["size"] == 8


def test_openai_compatibility_variants_share_deadline_and_attempt_budget(
    monkeypatch,
):
    clock = [0.0]
    calls = []
    client_timeouts = []

    class UnsupportedParameterError(Exception):
        status_code = 400

    class FakeCompletions:
        @staticmethod
        def create(**_kwargs):
            calls.append("http")
            clock[0] += 0.6
            if len(calls) == 1:
                raise UnsupportedParameterError(
                    "unsupported reasoning_effort parameter"
                )
            message = type("Message", (), {"content": '{"ok": true}'})()
            choice = type(
                "Choice",
                (),
                {"message": message, "finish_reason": "stop"},
            )()
            return type("Response", (), {"choices": [choice]})()

    class FakeClient:
        def __init__(self, **kwargs):
            client_timeouts.append(kwargs["timeout"])
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

        def close(self):
            pass

    import openai

    monkeypatch.setattr(openai, "OpenAI", FakeClient)
    monkeypatch.setattr(stage2_workflow.time, "monotonic", lambda: clock[0])
    config = PlainHandoffStage2Config(
        endpoint="http://stage2.test/v1",
        model="google/gemma-4-test",
        request_timeout=1.0,
        request_attempt_timeout=1.0,
        transport_max_attempts=3,
        transport_retry_backoff=0.0,
    )

    with pytest.raises(Stage2RequestExhaustedError, match="deadline"):
        stage2_workflow._completion_with_transport_retries(
            [{"role": "user", "content": "Return JSON."}],
            config,
            stage2_workflow._openai_completion,
        )

    assert calls == ["http", "http"]
    assert client_timeouts == pytest.approx([1.0, 0.4])

    calls.clear()
    clock[0] = 0.0

    class AlwaysUnsupported:
        @staticmethod
        def create(**_kwargs):
            calls.append("http")
            raise UnsupportedParameterError(
                "unsupported reasoning_effort parameter"
            )

    FakeCompletions.create = staticmethod(AlwaysUnsupported.create)
    budget_config = PlainHandoffStage2Config(
        endpoint="http://stage2.test/v1",
        model="google/gemma-4-test",
        request_timeout=10.0,
        request_attempt_timeout=10.0,
        transport_max_attempts=2,
        transport_retry_backoff=0.0,
    )
    with pytest.raises(Stage2RequestExhaustedError, match="attempt budget"):
        stage2_workflow._completion_with_transport_retries(
            [{"role": "user", "content": "Return JSON."}],
            budget_config,
            stage2_workflow._openai_completion,
        )
    assert calls == ["http", "http"]


def test_managed_vllm_pools_reject_overlapping_internal_ranges(
    tmp_path: Path,
    monkeypatch,
):
    primary = ManagedVLLMConfig(
        server_count=1,
        gpus=("cuda:0",),
        base_port=8010,
        internal_port_base=20_000,
    )
    extraction = ManagedVLLMConfig(
        server_count=1,
        gpus=("cuda:1",),
        base_port=9010,
        internal_port_base=20_000,
    )

    with pytest.raises(ValueError, match="overlapping internal rendezvous"):
        validate_managed_vllm_pool_isolation(primary, extraction)

    output_dir = tmp_path / "stage2"
    phase_path = output_dir / "vllm_servers" / "model_phase.json"
    phase_path.parent.mkdir(parents=True)
    phase_path.write_text(
        json.dumps(
            {
                "schema_version": stage2_workflow.MANAGED_MODEL_PHASE_SCHEMA_VERSION,
                "status": "running_configured_split",
                "active_role": "extraction",
                "allocation_mode": "configured_split",
                "transition": 2,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        stage2_workflow,
        "launch_managed_vllm_servers",
        lambda **_kwargs: pytest.fail(
            "overlapping pools must be rejected before either pool starts"
        ),
    )
    config = PlainHandoffStage2Config(
        endpoint="",
        model="primary-model",
        vllm=primary,
        vllm_rapid_switch_seconds=60.0,
        extraction_llm=Stage2ExtractionLLMConfig(
            model="extraction-model",
            vllm=extraction,
        ),
    )
    with pytest.raises(ValueError, match="overlapping internal rendezvous"):
        run_plain_handoff_stage2(
            handoff_path=tmp_path / "unused-handoff.jsonl",
            output_dir=output_dir,
            clinical_question="Identify confounders.",
            config=config,
            completion=lambda _messages, _config: "{}",
            extraction_completion=lambda _messages, _config: "{}",
            dataset=pd.DataFrame({"clinical_text": ["text"]}),
        )

    validate_managed_vllm_pool_isolation(
        primary,
        ManagedVLLMConfig(
            server_count=1,
            gpus=("cuda:1",),
            base_port=9010,
            internal_port_base=40_000,
        ),
    )


def test_logical_request_excludes_queue_time_and_keeps_slot_through_retries(monkeypatch):
    clock = [0.0]
    slot_entries = []
    slot_held = [False]
    calls = []
    events = []

    class BusySemaphore:
        def __enter__(self):
            assert not slot_held[0]
            # Waiting longer than the entire request budget must be harmless.
            clock[0] += 100.0
            slot_entries.append(clock[0])
            slot_held[0] = True

        def __exit__(self, *_args):
            slot_held[0] = False

    class TemporaryTransportError(Exception):
        pass

    def completion(messages, config):
        assert slot_held[0]
        calls.append((config.request_timeout, config.runtime_request_deadline))
        assert config.runtime_request_kind == "extraction"
        clock[0] += 2.0
        if len(calls) == 1:
            raise TemporaryTransportError("slow server")
        if len(calls) == 2:
            return "invalid JSON"
        assert "repair sentinel" in messages[-1]["content"]
        return '{"ok": true}'

    limiter = stage2_workflow._ConcurrencyLimitedCompletion(completion, 1)
    monkeypatch.setattr(limiter, "_semaphore", BusySemaphore())
    monkeypatch.setattr(stage2_workflow.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        stage2_workflow.time, "sleep", lambda delay: clock.__setitem__(0, clock[0] + delay)
    )
    monkeypatch.setattr(
        stage2_workflow, "_is_retryable_transport_error",
        lambda exc: isinstance(exc, TemporaryTransportError),
    )
    config = PlainHandoffStage2Config(
        endpoint="http://stage2.test/v1", model="test-model",
        request_timeout=10.0, request_attempt_timeout=4.0, transport_retry_backoff=1.0,
    )
    result = stage2_workflow._request_json(
        messages=[{"role": "user", "content": "Return JSON."}],
        config=config, completion=limiter, validate=dict,
        request_kind="extraction",
        repair_context={"allowed_feature_ids": ["repair sentinel"]},
        validation_event_observer=events.append,
    )
    assert result == {"ok": True}
    assert slot_entries == [100.0]
    assert calls == [(4.0, 110.0)] * 3
    assert clock[0] == 107.0
    assert not slot_held[0]
    assert len(events) == 1


def test_logical_request_releases_slot_on_deadline_failure(monkeypatch):
    clock = [0.0]
    calls = []

    class TemporaryTransportError(Exception):
        pass

    def completion(_messages, _config):
        calls.append("called")
        clock[0] += 11.0
        raise TemporaryTransportError("slow server")

    monkeypatch.setattr(stage2_workflow.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        stage2_workflow, "_is_retryable_transport_error",
        lambda exc: isinstance(exc, TemporaryTransportError),
    )
    limiter = stage2_workflow._ConcurrencyLimitedCompletion(completion, 1)
    config = PlainHandoffStage2Config(
        endpoint="http://stage2.test/v1", model="test-model", request_timeout=10.0,
    )
    with pytest.raises(Stage2RequestExhaustedError, match="deadline") as error:
        stage2_workflow._request_json(
            messages=[], config=config, completion=limiter, validate=dict,
        )
    assert isinstance(error.value.__cause__, TemporaryTransportError)
    assert calls == ["called"]
    assert limiter._semaphore.acquire(blocking=False)
    limiter._semaphore.release()


def test_leaf_checkpoint_survives_endpoint_and_runtime_budget_changes(tmp_path):
    from dataclasses import replace

    config = PlainHandoffStage2Config(
        endpoint="http://old-server.test/v1", model="same-model", workers=32,
    )
    kwargs = dict(
        output_dir=tmp_path / "initial",
        input_value={"phase": "initial_interpretation", "packets": []},
        messages=[{"role": "user", "content": "Return JSON."}],
        validate=dict,
    )
    result = stage2_workflow._checkpointed_request_json(
        **kwargs, config=config, completion=lambda *_args: '{"ok": true}',
    )

    def should_not_call(*_args):
        pytest.fail("A server/runtime-only change must reuse the saved LLM response")

    moved = replace(
        config, endpoint="http://new-server.test/v1", workers=4,
        request_timeout=2700.0, request_attempt_timeout=900.0,
    )
    assert stage2_workflow._checkpointed_request_json(
        **kwargs, config=moved, completion=should_not_call,
    ) == result
