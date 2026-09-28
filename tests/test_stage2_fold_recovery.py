import json
import threading
from collections import Counter
from dataclasses import replace

import pytest

import oci.inference.plain_handoff_stage2 as stage2
from oci.inference.plain_handoff_stage2_analysis import Stage2ResponseValidationError


def runner(**options):
    return stage2.PlainHandoffStage2(
        config=stage2.PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1", model="test-model",
            required_architectures=(), **options,
        ), clinical_question="Identify pretreatment variables.",
        completion=lambda *_: '{"ok": true}',
    )


def test_run_recovers_fold_without_repeating_completed_requests_or_blocking_sibling(tmp_path, monkeypatch):
    task = runner(workers=2)
    attempts, requests = Counter(), []
    sibling_finished = threading.Event()
    output = tmp_path / "stage2"
    monkeypatch.setattr(task, "_load_or_compile_evidence", lambda **_: (
        [{"outer_fold": 1}, {"outer_fold": 2}], {},
    ))

    def complete(*_):
        requests.append("completed request")
        return '{"ok": true}'

    def fold(**kwargs):
        index = kwargs["outer_fold"]
        attempts[index] += 1
        if index == 1:
            stage2._checkpointed_request_json(
                output_dir=kwargs["output_dir"] / "saved_request",
                input_value={"phase": "test"},
                messages=[{"role": "user", "content": "Return JSON."}],
                config=task.config, completion=complete, validate=dict,
            )
            if attempts[index] == 1:
                raise stage2.Stage2RequestExhaustedError("endpoint temporarily unavailable")
        else:
            sibling_finished.set()
        return {"outer_fold": index, "features": [], "review_converged": True}

    def backoff(delay):
        assert delay == 60
        assert sibling_finished.wait(timeout=2), "Recovery blocked the sibling fold"
        state = json.loads((output / "outer_001/recovery_status.json").read_text())
        assert state["status"] == "retry_wait"

    monkeypatch.setattr(task, "_run_outer_fold", fold)
    monkeypatch.setattr(stage2.time, "sleep", backoff)
    result = task.run(handoff_path=tmp_path / "unused.jsonl", output_dir=output)
    assert result["outer_folds"] == 2
    assert attempts == {1: 2, 2: 1}
    assert requests == ["completed request"]
    state = json.loads((output / "outer_001/recovery_status.json").read_text())
    assert (state["status"], state["recoveries_used"]) == ("complete", 1)
    events = [json.loads(line) for line in (output / "outer_001/recovery_events.jsonl").read_text().splitlines()]
    assert [event["status"] for event in events] == ["running", "retry_wait", "running", "complete"]


@pytest.mark.parametrize("limit", [0, 2])
def test_recovery_exhaustion_is_bounded_and_preserves_the_error(tmp_path, monkeypatch, limit):
    task = runner(outer_fold_recovery_attempts=limit)
    error = stage2.Stage2RequestExhaustedError("endpoint still unavailable")
    attempts, waits = [], []

    def fail(**_):
        attempts.append(1)
        raise error

    monkeypatch.setattr(task, "_run_outer_fold", fail)
    monkeypatch.setattr(stage2.time, "sleep", waits.append)
    with pytest.raises(stage2.Stage2RequestExhaustedError) as caught:
        task._run_outer_fold_with_recovery(outer_fold=1, output_dir=tmp_path)
    assert caught.value is error
    assert len(attempts) == limit + 1
    assert waits == [60.0 * attempt for attempt in range(1, limit + 1)]
    state = json.loads((tmp_path / "recovery_status.json").read_text())
    assert state["status"] == "failed" and state["recoveries_used"] == limit


@pytest.mark.parametrize("error", [ValueError("bad config"), Stage2ResponseValidationError("invalid answer"), PermissionError("checkpoint inaccessible")])
def test_recovery_does_not_hide_nontransport_failures(tmp_path, monkeypatch, error):
    task = runner()
    attempts = []

    def fail(**_):
        attempts.append(1)
        raise error

    monkeypatch.setattr(task, "_run_outer_fold", fail)
    monkeypatch.setattr(stage2.time, "sleep", lambda _: pytest.fail("Unexpected retry"))
    with pytest.raises(type(error)) as caught:
        task._run_outer_fold_with_recovery(outer_fold=1, output_dir=tmp_path)
    assert caught.value is error and len(attempts) == 1


def test_operational_recovery_policy_roundtrips_without_changing_scientific_identity():
    original = runner().config
    raw = {**original.public_dict(), "request_timeout": 18000,
           "request_attempt_timeout": 5400, "outer_fold_recovery_attempts": 3,
           "outer_fold_recovery_backoff": 120}
    changed = stage2.plain_stage2_config_from_mapping(raw, default_workers=8)
    assert changed.outer_fold_recovery_attempts == 3
    assert changed.outer_fold_recovery_backoff == 120
    arguments = dict(clinical_question="Question", outer_fold=1, discovery_packets=[], seed=42)
    assert stage2._feature_definition_input_value(config=original, **arguments) == stage2._feature_definition_input_value(config=changed, **arguments)
    for bad in [True, -1, 1.5, "2"]:
        with pytest.raises(ValueError, match="outer_fold_recovery_attempts"):
            replace(original, outer_fold_recovery_attempts=bad).validate()
    for bad in [True, -1, float("inf"), "60"]:
        with pytest.raises(ValueError, match="outer_fold_recovery_backoff"):
            replace(original, outer_fold_recovery_backoff=bad).validate()
