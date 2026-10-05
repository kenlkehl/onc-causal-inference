from contextlib import contextmanager
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading

import pandas as pd
import pytest

from oci.inference import plain_handoff_stage2 as workflow
from oci.inference import plain_handoff_stage2_analysis as analysis
from oci.inference import stage2_decision_ontology as ontology
from oci.inference.stage2_decision_config import DecisionExtractionConfig
from oci.inference.stage2_parallel import ordered_map
from oci.inference.vllm_server_pool import (
    ManagedVLLMConfig, co_resident_vllm_configs, validate_managed_vllm_pool_isolation,
)


def pool(gpus, *, port=8010, fraction="0.50", width=1, internal=20000):
    return ManagedVLLMConfig(server_count=len(gpus)//width,
        gpus=tuple(f"cuda:{gpu}" for gpu in gpus), gpus_per_server=width,
        base_port=port, internal_port_base=internal,
        extra_args=("--gpu-memory-utilization", fraction, "--max-num-seqs", "4"))


@pytest.mark.parametrize("gpus,width", [([0], 1), ([2, 5], 1), ([0, 1, 2, 3, 4], 1), (list(range(8)), 2)])
def test_co_resident_allocation_preserves_gpu_union_widths_and_budgets(gpus, width):
    primary = pool(gpus[:width], width=width)
    extractor = pool(gpus, port=8011, fraction="0.28", internal=20128)
    before = primary, extractor
    left, right = co_resident_vllm_configs(primary, extractor)
    assert left.gpus == right.gpus == tuple(f"cuda:{gpu}" for gpu in gpus)
    assert left.effective_gpus_per_server() == width
    assert right.effective_gpus_per_server() == 1
    assert left.server_count == len(gpus)//width
    assert left.extra_args == primary.extra_args and right.extra_args == extractor.extra_args
    assert not set(left.effective_ports()) & set(right.effective_ports())
    validate_managed_vllm_pool_isolation(left, right)
    assert co_resident_vllm_configs(left, right) == (left, right)
    assert before == (primary, extractor)


@pytest.mark.parametrize("fraction", ["0.5", "nan", "inf", "-1", "0", "invalid"])
def test_co_resident_allocation_rejects_invalid_memory_budgets(fraction):
    with pytest.raises(ValueError):
        co_resident_vllm_configs(pool([0]), pool([1], port=8110, fraction=fraction))


def test_co_resident_allocation_requires_explicit_budgets_and_compatible_widths():
    with pytest.raises(ValueError, match="explicit"):
        co_resident_vllm_configs(replace(pool([0]), extra_args=()), pool([1], port=8110))
    with pytest.raises(ValueError, match="tensor-parallel"):
        co_resident_vllm_configs(pool([0, 1], width=2), pool([2], port=8110, fraction="0.28"))
    with pytest.raises(ValueError, match="HTTP ports"):
        co_resident_vllm_configs(pool([0]), pool([1], fraction="0.28"))
    left, right = co_resident_vllm_configs(
        replace(pool([0]), extra_args=("--gpu-memory-utilization=0.5",)),
        replace(pool([1], port=8110), extra_args=("--gpu-memory-utilization=0.28",)))
    assert left.server_count == right.server_count == 2


def test_runtime_allocation_and_worker_count_preserve_feature_definition_identity():
    before = workflow.PlainHandoffStage2Config(endpoint="", model="review-model", vllm=pool([0]),
        extraction_llm=workflow.Stage2ExtractionLLMConfig(endpoint="", model="extract-model",
            vllm=pool([0, 1], port=8110, fraction="0.28", internal=30000)))
    after = replace(before, vllm_co_resident_all_gpus=True, workers=32)
    after.validate(require_endpoint=False)
    args = dict(clinical_question="test", outer_fold=1, discovery_packets=[], seed=42)
    assert workflow._feature_definition_input_value(config=before, **args) == workflow._feature_definition_input_value(config=after, **args)


@pytest.mark.parametrize("decision,disabled", [(True, False), (False, False), (False, True)])
def test_co_resident_models_remain_resident_for_all_folds(tmp_path, monkeypatch, decision, disabled):
    config = workflow.PlainHandoffStage2Config(endpoint="", model="review-model",
        vllm=pool([2]), vllm_co_resident_all_gpus=True,
        extraction_llm=workflow.Stage2ExtractionLLMConfig(endpoint="", model="extract-model",
            vllm=pool([2, 5, 7], port=8110, fraction="0.28", internal=30000)),
        decision_extraction=DecisionExtractionConfig(enabled=decision), runtime_disable_extraction=disabled)
    lifecycle, allocations = [], {}

    @contextmanager
    def launch(**kwargs):
        role = kwargs["output_dir"].name
        lifecycle.append("start-" + role)
        cfg = kwargs["config"]
        allocations[role] = cfg
        assert cfg.gpus == ("cuda:2", "cuda:5", "cuda:7")
        assert cfg.server_count == 3
        try:
            yield tuple(f"http://localhost:{p}/v1" for p in cfg.effective_ports())
        finally:
            lifecycle.append("stop-" + role)

    def run(self, **kwargs):
        assert len(self.config.runtime_endpoints) == 3
        assert lifecycle == (["start-orchestrator"] if disabled else ["start-orchestrator", "start-extractor"])
        if not disabled:
            assert len(self.config.extraction_llm.runtime_endpoints) == 3
        return {"completed": True}

    monkeypatch.setattr(workflow, "launch_managed_vllm_servers", launch)
    monkeypatch.setattr(workflow.PlainHandoffStage2, "run", run)
    monkeypatch.setattr(workflow, "_served_model_ids", lambda c: [c.model])
    # A completed preparation barrier avoids triggering initial ontology work.
    (tmp_path / "handoff.jsonl").write_text(json.dumps({"outer_fold": 1}) + "\n")
    if decision:
        folder = tmp_path / "stage2/outer_001"
        folder.mkdir(parents=True)
        feature = definition("color")
        (folder / "feature_definitions.json").write_text(json.dumps({"features": [feature]}))
        ontology.prepare_ontologies(analysis.initial_feature_modeling_definitions([feature]),
            output_dir=folder / "ontology_supervision/round_001/failure_ontology_refinement",
            request_json=lambda *a, **k: pytest.fail("no ontology preparation needed"))
    result = workflow.run_plain_handoff_stage2(handoff_path=tmp_path / "handoff.jsonl",
        output_dir=tmp_path / "stage2", clinical_question="test", config=config, dataset=object())
    assert result == {"completed": True}
    assert lifecycle == (["start-orchestrator", "stop-orchestrator"] if disabled else
        ["start-orchestrator", "start-extractor", "stop-extractor", "stop-orchestrator"])
    if not disabled:
        phase = json.loads((tmp_path / "stage2/vllm_servers/model_phase.json").read_text())
        assert phase["allocation_mode"] == "co_resident_all_gpus"
        assert phase["status"] == "complete"


def test_ordered_map_is_bounded_parallel_and_preserves_order():
    entered = threading.Event()
    release = threading.Event()
    lock = threading.Lock()
    produced, running, peak = [], 0, 0

    def source():
        for i in range(12):
            produced.append(i)
            yield i

    def work(i):
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
            if running == 3:
                entered.set()
        assert release.wait(5)
        with lock:
            running -= 1
        return i * 2

    with ThreadPoolExecutor(max_workers=1) as caller:
        future = caller.submit(ordered_map, work, source(), workers=3)
        try:
            assert entered.wait(5)
            assert produced == [0, 1, 2]
        finally:
            release.set()
        assert future.result() == [i*2 for i in range(12)]
    assert peak == 3


def definition(name):
    return {"feature_id": name, "name": name, "description": "Recorded status",
        "value_type": "categorical", "categories_or_unit": ["red", "blue"],
        "measurement_definition": "Record the pretreatment status", "missing_value_rule": "Null when absent"}


def test_parallel_ontology_revisions_match_serial_and_reuse_checkpoints(tmp_path):
    features = [definition(f"feature_{i}") for i in range(3)]
    features += [{**definition("locked"), "configured_explicit_feature": True}, definition("untriggered")]
    summary = {"feature_failure_patterns": [{"feature_name": f["name"], "failure_kind": "none_of_above",
        "patient_count": 10, "patient_fraction": 0.3, "patient_row_ids": list(range(10))} for f in features[:-1]]}
    barrier = threading.Barrier(3)
    calls = []

    def answer(messages, validate, *, request_kind):
        assert request_kind == "interpretation"
        payload = json.loads(messages[1]["content"])
        calls.append(payload["feature"]["name"])
        barrier.wait(timeout=5)
        return validate({"action": "revise", "rationale": "Add a supported category",
                         "categories_or_unit": ["red", "blue", "green"]})

    args = dict(summary=summary, policy=DecisionExtractionConfig(), extraction_dir=tmp_path / "empty",
                output_dir=tmp_path / "parallel", max_prompt_chars=100000)
    parallel = ontology.revise_ontologies(features, request_json=answer, workers=3, **args)
    assert set(calls) == {f"feature_{i}" for i in range(3)}
    assert parallel[1] == [f"feature_{i}" for i in range(3)]
    assert [f["name"] for f in parallel[0]] == [f["name"] for f in features]
    assert features[0]["categories_or_unit"] == ["red", "blue"]
    restored = ontology.revise_ontologies(features, request_json=lambda *a, **k: pytest.fail("checkpoint reused"),
                                          workers=2, **args)
    assert restored == parallel
    serial = ontology.revise_ontologies(features, request_json=lambda m, v, **k: v({"action": "revise",
        "rationale": "Add a supported category", "categories_or_unit": ["red", "blue", "green"]}),
        workers=1, **{**args, "output_dir": tmp_path / "serial"})
    assert serial == parallel


def test_failed_parallel_revision_preserves_successful_feature_checkpoints(tmp_path):
    features = [definition(f"feature_{i}") for i in range(3)]
    summary = {"feature_failure_patterns": [{"feature_name": f["name"], "failure_kind": "none_of_above",
        "patient_count": 10, "patient_fraction": 0.3} for f in features]}
    barrier = threading.Barrier(3)

    def request(messages, validate, **kwargs):
        name = json.loads(messages[1]["content"])["feature"]["name"]
        barrier.wait(timeout=5)
        if name == "feature_1":
            raise ValueError("invalid review")
        return validate({"action": "keep", "rationale": "Keep the declared categories"})

    args = dict(summary=summary, policy=DecisionExtractionConfig(), extraction_dir=tmp_path / "empty",
                output_dir=tmp_path / "reviews", max_prompt_chars=100000, workers=3)
    with pytest.raises(ValueError, match="invalid review"):
        ontology.revise_ontologies(features, request_json=request, **args)
    assert not (args["output_dir"] / "result.json").exists()
    assert len(list(args["output_dir"].rglob("response.json"))) == 2
    resumed = []

    def recover(messages, validate, **kwargs):
        resumed.append(json.loads(messages[1]["content"])["feature"]["name"])
        return validate({"action": "keep", "rationale": "Keep the declared categories"})

    updated, changed, report = ontology.revise_ontologies(features, request_json=recover, **args)
    assert resumed == ["feature_1"]
    assert updated == features and changed == []


def test_shared_examples_scan_preserves_sampling_order_and_budgets(tmp_path, monkeypatch):
    for i, folder in enumerate(("a", "a.x", "b", "c")):
        for name in ("first", "second"):
            path = tmp_path / "decisions" / folder / name / "result.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"feature_name": name, "status": "none_of_above", "calls": [
                {"messages": [{}, {"content": json.dumps({"evidence": f"evidence-{folder}-{name}"})}]}]}))
    expected = {}
    budgets = {"first": 10000, "second": 25}
    for name, budget in budgets.items():
        values, used = [], 0
        for path in sorted(tmp_path.rglob("result.json")):
            r = json.loads(path.read_text())
            if r["feature_name"] != name:
                continue
            e = json.loads(r["calls"][-1]["messages"][1]["content"])["evidence"]
            if e not in values and used + len(json.dumps(e)) <= budget:
                values.append(e)
                used += len(json.dumps(e))
            if len(values) == 3:
                break
        expected[name] = values
    read, original = [], Path.read_text

    def track(path, *a, **kw):
        read.append(str(path))
        return original(path, *a, **kw)

    monkeypatch.setattr(Path, "read_text", track)
    assert ontology._examples_for_features(tmp_path, budgets) == expected
    assert len(read) == len(set(read))


def test_parallel_harmonization_preserves_serial_results_and_checkpoints(tmp_path):
    names = [f"feature_{i}" for i in range(3)]
    features = [{**definition(name), "value_type": "continuous", "categories_or_unit": ["unitless"]} for name in names]
    frame = pd.DataFrame({name: [1.0, "unknown", 3.0] for name in names})
    barrier = threading.Barrier(3)

    def request(messages, validate, **kwargs):
        barrier.wait(timeout=5)
        return validate({"target_representation": "continuous", "canonical_categories": [], "numeric_bin_rules": [],
            "categorical_value_map": [{"raw_value": "unknown", "canonical_value": None}],
            "reason": "Unknown has no numeric value"})

    args = dict(extracted=frame, definitions=features, output_dir=tmp_path / "parallel", max_prompt_chars=100000)
    parallel = analysis._harmonize_training_extraction(request_json=request, workers=3, **args)
    restored = analysis._harmonize_training_extraction(request_json=lambda *a, **k: pytest.fail("reused"),
        workers=2, **args)
    pd.testing.assert_frame_equal(parallel[0], restored[0])
    assert parallel[1] == restored[1]
    assert parallel[2]["features_requested_from_llm"] == names
    pd.testing.assert_frame_equal(frame, pd.DataFrame({name: [1.0, "unknown", 3.0] for name in names}))


def test_managed_primary_router_enforces_shared_and_per_server_limits(monkeypatch):
    cfg = workflow.PlainHandoffStage2Config(endpoint="http://localhost:8010/v1", model="review-model",
        vllm=pool([0, 1, 2]), workers=4,
        runtime_endpoints=tuple(f"http://localhost:{8010+i}/v1" for i in range(3)))
    monkeypatch.setattr(workflow, "_served_model_ids", lambda c: [c.model])
    release, full = threading.Event(), threading.Event()
    lock, active, peaks = threading.Lock(), {}, {}
    total_peak = 0

    def transport(messages, config):
        nonlocal total_peak
        url = config.endpoint
        with lock:
            active[url] = active.get(url, 0)+1
            peaks[url] = max(peaks.get(url, 0), active[url])
            total_peak = max(total_peak, sum(active.values()))
            if sum(active.values()) == 4:
                full.set()
        assert release.wait(5)
        with lock:
            active[url] -= 1
        return "{}"

    monkeypatch.setattr(workflow, "_openai_completion", transport)
    runner = workflow.PlainHandoffStage2(config=cfg, clinical_question="test")
    router = runner.completion.completion
    assert isinstance(router, workflow._LoadAwareOpenAICompletion)
    router.pool._metrics_reader = None
    with ThreadPoolExecutor(max_workers=16) as callers:
        futures = [callers.submit(runner.completion, [], cfg) for _ in range(16)]
        try:
            assert full.wait(5)
        finally:
            release.set()
        assert [f.result() for f in futures] == ["{}"]*16
    assert total_peak == 4
    assert len(peaks) == 3
    assert max(peaks.values()) <= 2
