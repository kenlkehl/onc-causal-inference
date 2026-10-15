from concurrent.futures import ThreadPoolExecutor
import threading

import pandas as pd
import pytest

from oci.inference import plain_handoff_stage2 as workflow
from oci.inference import plain_handoff_stage2_analysis as analysis
from oci.inference.stage2_parallel import ordered_map
from oci.inference.vllm_server_pool import ManagedVLLMConfig


def pool(gpus, *, port=8010, fraction="0.50", width=1, internal=20000):
    return ManagedVLLMConfig(server_count=len(gpus)//width,
        gpus=tuple(f"cuda:{gpu}" for gpu in gpus), gpus_per_server=width,
        base_port=port, internal_port_base=internal,
        extra_args=("--gpu-memory-utilization", fraction, "--max-num-seqs", "4"))


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
