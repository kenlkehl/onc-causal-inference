from __future__ import annotations

import json
import os
import signal
from pathlib import Path

import pytest

import oci.inference.vllm_server_pool as server_pool
from oci.inference.plain_handoff_stage2 import (
    plain_stage2_config_from_mapping,
)
from oci.inference.vllm_server_pool import (
    ManagedVLLMConfig,
    ManagedVLLMServerPool,
)


@pytest.mark.parametrize(
    ("vllm", "message"),
    [
        (
            {"server_count": 3, "gpus": [0, 1, 2, 3]},
            "divide evenly",
        ),
    ],
)
def test_managed_vllm_rejects_unsafe_or_ambiguous_layouts(vllm, message):
    with pytest.raises(ValueError, match=message):
        plain_stage2_config_from_mapping(
            {"model": "Qwen/Qwen3.8-27B", "vllm": vllm},
            default_workers=4,
        )


def test_managed_pool_launches_one_process_per_gpu_and_writes_a_redacted_manifest(
    tmp_path,
    monkeypatch,
):
    popen_calls = []

    class FakeProcess:
        next_pid = 10_000

        def __init__(self, command, **kwargs):
            self.command = list(command)
            self.kwargs = kwargs
            self.pid = FakeProcess.next_pid
            FakeProcess.next_pid += 1
            self.returncode = None
            popen_calls.append(self)

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -signal.SIGTERM

        def kill(self):
            self.returncode = -signal.SIGKILL

        def wait(self, timeout=None):
            assert timeout is not None
            if self.returncode is None:
                self.returncode = 0
            return self.returncode

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,6")
    monkeypatch.setenv("VLLM_PORT", "37973")
    monkeypatch.setattr(server_pool, "_assert_port_available", lambda *_args: None)
    monkeypatch.setattr(
        server_pool,
        "_server_model_ids",
        lambda *_args, **_kwargs: ["Qwen/Qwen3.8-27B"],
    )
    monkeypatch.setattr(server_pool.subprocess, "Popen", FakeProcess)

    def fake_signal(server, sig):
        if server.process.poll() is not None:
            return
        if sig == signal.SIGTERM:
            server.process.terminate()
        else:
            server.process.kill()

    monkeypatch.setattr(
        ManagedVLLMServerPool,
        "_signal_process_group",
        staticmethod(fake_signal),
    )
    config = ManagedVLLMConfig(
        server_count=2,
        gpus=("cuda:0", "cuda:1"),
        reasoning_parser="qwen3",
        language_model_only=True,
    )
    pool = ManagedVLLMServerPool(
        config=config,
        model="Qwen/Qwen3.8-27B",
        api_key="super-secret",
        output_dir=tmp_path / "servers",
    )

    assert pool.start() == (
        "http://127.0.0.1:8010/v1",
        "http://127.0.0.1:8011/v1",
    )
    assert [call.kwargs["env"]["CUDA_VISIBLE_DEVICES"] for call in popen_calls] == [
        "4",
        "6",
    ]
    assert [call.kwargs["env"]["VLLM_PORT"] for call in popen_calls] == [
        "20000",
        "20128",
    ]
    interpreter_bin = str(Path(server_pool.sys.executable).parent)
    assert all(
        call.kwargs["env"]["PATH"].split(os.pathsep)[0] == interpreter_bin
        for call in popen_calls
    )
    assert all("--tensor-parallel-size" in call.command for call in popen_calls)
    assert all(
        call.command[call.command.index("--tensor-parallel-size") + 1] == "1"
        for call in popen_calls
    )

    pool.stop()

    manifest_text = (tmp_path / "servers" / "manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert manifest["status"] == "stopped"
    assert [server["exit_code"] for server in manifest["servers"]] == [
        -signal.SIGTERM,
        -signal.SIGTERM,
    ]
    assert "super-secret" not in manifest_text
    assert "<redacted>" in manifest_text
    assert [server["vllm_internal_port_base"] for server in manifest["servers"]] == [
        20_000,
        20_128,
    ]
    assert all(call.kwargs["stdout"].closed for call in popen_calls)
