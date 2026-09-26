from concurrent.futures import ThreadPoolExecutor
import json
import threading

import pytest

from oci.inference.stage2_concurrency import LiveRequestLimits


def _policy(path, interpretation, extraction, total):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(dict(
        interpretation=interpretation, extraction=extraction, total=total,
    )))
    temporary.replace(path)


def test_live_total_limit_drains_both_roles_and_resumes_without_cancelling(tmp_path):
    path = tmp_path / "limits.json"
    _policy(path, 2, 2, 2)
    limits = LiveRequestLimits(path, ceilings=dict(
        interpretation=2, extraction=2, total=4,
    ), poll_seconds=0.01)
    started, admitted = threading.Event(), threading.Event()

    def waiting_request():
        started.set()
        with limits.slot("extraction"):
            admitted.set()

    with ThreadPoolExecutor(max_workers=1) as executor:
        with limits.slot("interpretation"):
            with limits.slot("extraction"):
                _policy(path, 2, 2, 1)
                future = executor.submit(waiting_request)
                assert started.wait(2)
                assert not admitted.wait(0.05)
            # One remaining active request still fills the lowered total limit.
            assert not admitted.wait(0.05)
            _policy(path, 2, 2, 2)
            assert admitted.wait(2)
        future.result(timeout=2)


def test_live_role_limit_and_exception_release(tmp_path):
    path = tmp_path / "limits.json"
    _policy(path, 1, 1, 2)
    limits = LiveRequestLimits(path, ceilings=dict(
        interpretation=2, extraction=2, total=4,
    ), poll_seconds=0.01)
    admitted = threading.Event()

    def waiting_request():
        with limits.slot("interpretation"):
            admitted.set()

    with ThreadPoolExecutor(max_workers=1) as executor:
        with pytest.raises(RuntimeError, match="request failed"):
            with limits.slot("interpretation"):
                future = executor.submit(waiting_request)
                assert not admitted.wait(0.05)
                # The other role remains usable while interpretation is full.
                with limits.slot("extraction"):
                    pass
                raise RuntimeError("request failed")
        assert admitted.wait(2)
        future.result(timeout=2)


def test_invalid_live_edit_keeps_last_limit_and_zero_pauses_admission(tmp_path, caplog):
    path = tmp_path / "limits.json"
    _policy(path, 1, 1, 0)
    limits = LiveRequestLimits(path, ceilings=dict(
        interpretation=2, extraction=2, total=4,
    ), poll_seconds=0.01)
    path.write_text('{"total": 1000}')
    admitted = threading.Event()

    def waiting_request():
        with limits.slot("interpretation"):
            admitted.set()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(waiting_request)
        assert not admitted.wait(0.05)
        _policy(path, 1, 1, 1)
        assert admitted.wait(2)
        future.result(timeout=2)
    assert "Keeping last valid" in caplog.text
