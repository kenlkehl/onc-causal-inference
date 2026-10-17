"""Real serial/parallel equivalence, global capacity and checkpoint reuse."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import os
import subprocess
import sys
import textwrap
import time

import pytest

from oci.inference import stage2_modeling_pool as pool
from oci.inference import stage2_multi_model_selection as numerical
from oci.inference import stage2_modifier_count as count
from oci.inference.stage2_multi_model_config import ModifierCountConfig
from oci.inference.stage2_role_adjudication import Stage2RoleAdjudicationConfig
from tests.test_stage2_multi_model import sample_inputs


def _timed_job(value):
    start = time.monotonic()
    time.sleep(0.08)
    return value, os.getpid(), start, time.monotonic(), pool.enabled()


def test_modeling_pool_does_not_require_joblib_vendored_cloudpickle():
    # Initialize joblib first: older releases legitimately use their vendored
    # copy internally, but the modeling pool must not depend on that layout.
    script = textwrap.dedent("""
        import builtins
        import joblib

        original_import = builtins.__import__

        def without_vendored_cloudpickle(name, globals=None, locals=None, fromlist=(), level=0):
            if name == 'joblib.externals' and 'cloudpickle' in (fromlist or ()):
                raise ImportError('joblib does not vendor cloudpickle')
            return original_import(name, globals, locals, fromlist, level)

        builtins.__import__ = without_vendored_cloudpickle
        from oci.inference import stage2_modeling_pool as pool

        def make_job(value):
            return lambda: value + 1

        payload = pool.cloudpickle.dumps(make_job(41))
        assert pool._execute(payload) == 42
    """)
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_global_pool_shares_budget_freezes_closures_and_disables_nested_pools():
    def producer(offset):
        pending = []
        for index in range(6):
            pending.append(pool.submit(lambda: _timed_job(offset + index)))
        return [f.result() for f in pending]

    with pool.modeling_session(2):
        with ThreadPoolExecutor(max_workers=3) as producers:
            rows = sum(list(producers.map(producer, [0, 10, 20])), [])
    assert not pool.enabled()
    assert sorted(r[0] for r in rows) == [*range(6), *range(10, 16), *range(20, 26)]
    assert len({r[1] for r in rows}) == 2
    assert all(not r[4] for r in rows)
    events = sorted([(r[2], 1) for r in rows] + [(r[3], -1) for r in rows])
    active, peak = 0, 0
    for _, delta in events:
        active += delta
        peak = max(peak, active)
    assert peak == 2


def test_checkpoint_resume_corruption_and_worker_exception(tmp_path):
    with pool.modeling_session(2):
        future = numerical.checkpoint_future(tmp_path, "cell", "fp", lambda: {"value": 12})
        assert future.result() == {"value": 12}
        def forbidden():
            raise AssertionError("completed checkpoint must not be fitted again")
        assert numerical.checkpoint_future(tmp_path, "cell", "fp", forbidden).result() == {"value": 12}
        with pytest.raises(ValueError, match="incompatible"):
            numerical.checkpoint_future(tmp_path, "cell", "wrong", forbidden)
        (tmp_path / "cell.json").write_text(json.dumps({"input_fingerprint": "fp", "result": {}}))
        with pytest.raises(ValueError, match="corrupt"):
            numerical.checkpoint_future(tmp_path, "cell", "fp", forbidden)
        with pytest.raises(ZeroDivisionError):
            numerical.checkpoint_future(tmp_path, "failed", "fp", lambda: 1 / 0).result()
        assert not (tmp_path / "failed.json").exists()
        assert pool.call(lambda: 7) == 7


@pytest.mark.parametrize("binary", [False, True])
def test_real_evidence_and_model_selection_identical_in_parallel(tmp_path, binary):
    arguments = sample_inputs()
    if binary:
        arguments["dataset"]["outcome"] = (arguments["dataset"]["outcome"] > 0).astype(int)
        arguments["outcome_type"] = "binary"
    arguments["policy"] = replace(arguments["policy"], multi_model=replace(
        arguments["policy"].multi_model,
        stability=replace(arguments["policy"].multi_model.stability, partition_repeats=2, subset_partitions=2),
        modifier_count=ModifierCountConfig(candidate_counts=(0, 1), max_ranked_modifiers=2, forest_seeds=2),
    ))
    selected = [{**f, "roles": ["confounder", "effect_modifier"]} for f in arguments["definitions"]]
    roles = {"decisions": [{"feature_id": f["feature_id"], "roles": f["roles"]} for f in selected]}
    results = []
    def no_llm(*args, **kwargs):
        raise AssertionError("numerical ranking must not call an LLM")
    for workers in [1, 2]:
        root = tmp_path / str(workers)
        with pool.modeling_session(workers):
            evidence = numerical.select_stage2_features_multi_model(**arguments, checkpoint_dir=root / "evidence")
            result = count.select_modifier_count(
                **arguments, statistical_report=evidence[1], selected=selected, role_report=roles,
                request_json=no_llm, role_policy=Stage2RoleAdjudicationConfig(),
                output_dir=root / "count", numerical_checkpoint_dir=root / "evidence",
                model_identity="test", estimation_trees=12,
            )
        # Normalize the operational path included in each selection fold audit.
        value = json.dumps([evidence, result], sort_keys=True).replace(str(root), "ROOT")
        results.append(value)
    assert results[0] == results[1]


def test_config_and_serial_session_validation():
    from oci.inference.plain_handoff_stage2 import PlainHandoffStage2Config, plain_stage2_config_from_mapping
    raw = {"endpoint": "http://example.test/v1", "model": "test", "workers": 8, "modeling_workers": 3}
    parsed = plain_stage2_config_from_mapping(raw, default_workers=1)
    assert parsed.modeling_workers == 3 and parsed.workers == 8
    assert plain_stage2_config_from_mapping(parsed.public_dict(), default_workers=1).modeling_workers == 3
    default = plain_stage2_config_from_mapping({k: v for k, v in raw.items() if k != "modeling_workers"}, default_workers=1)
    assert default.modeling_workers == 1
    assert parsed.statistical_selection.public_dict() == default.statistical_selection.public_dict()
    for value in [0, -1, True, "4"]:
        with pytest.raises(ValueError, match="modeling_workers"):
            with pool.modeling_session(value):
                pass
        config = PlainHandoffStage2Config(endpoint="http://example.test/v1", modeling_workers=value)
        with pytest.raises(ValueError, match="modeling_workers"):
            config.validate(require_model=False)
    with pool.modeling_session(1):
        assert not pool.enabled()
        assert pool.call(os.getpid) == os.getpid()
