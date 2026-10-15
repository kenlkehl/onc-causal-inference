from __future__ import annotations

import sys

import pytest

from scripts import detect_all_evidence_hardware as hardware


def test_hardware_detection_selects_eligible_gpus_and_sizes_workers(monkeypatch, capsys):
    monkeypatch.setattr(
        hardware,
        "_visible_gpus",
        lambda: [
            hardware.GPU(index=0, name="GPU A", free_gib=48.0, total_gib=48.0),
            hardware.GPU(index=1, name="GPU B", free_gib=12.0, total_gib=24.0),
            hardware.GPU(index=2, name="GPU C", free_gib=32.0, total_gib=48.0),
        ],
    )
    monkeypatch.setattr(hardware, "_available_cpu_count", lambda: 64)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "detect_all_evidence_hardware.py",
            "--gpu-count",
            "auto",
            "--workers",
            "auto",
            "--stage2-workers",
            "auto",
            "--outer-folds",
            "5",
            "--inner-folds",
            "5",
            "--min-free-vram-gib",
            "20",
        ],
    )

    assert hardware.main() == 0

    fields = capsys.readouterr().out.strip().split("\t")
    assert fields[:5] == ["2", "cuda:0,cuda:2", "30", "32", "64"]
    assert "cuda:0 GPU A 48.0/48.0 GiB free" in fields[5]
    assert "cuda:2 GPU C 32.0/48.0 GiB free" in fields[5]


def test_hardware_detection_honors_gpu_and_worker_limits(monkeypatch, capsys):
    monkeypatch.setattr(
        hardware,
        "_visible_gpus",
        lambda: [
            hardware.GPU(index=0, name="smaller", free_gib=24.0, total_gib=48.0),
            hardware.GPU(index=1, name="larger", free_gib=40.0, total_gib=48.0),
        ],
    )
    monkeypatch.setattr(hardware, "_available_cpu_count", lambda: 16)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "detect_all_evidence_hardware.py",
            "--gpu-count",
            "1",
            "--workers",
            "7",
            "--stage2-workers",
            "5",
            "--outer-folds",
            "5",
            "--inner-folds",
            "5",
        ],
    )

    assert hardware.main() == 0

    fields = capsys.readouterr().out.strip().split("\t")
    assert fields[:5] == ["1", "cuda:1", "7", "5", "16"]


def test_hardware_detection_fails_when_no_gpu_has_enough_free_vram(
    monkeypatch,
):
    monkeypatch.setattr(
        hardware,
        "_visible_gpus",
        lambda: [hardware.GPU(index=0, name="busy", free_gib=4.0, total_gib=48.0)],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "detect_all_evidence_hardware.py",
            "--outer-folds",
            "5",
            "--inner-folds",
            "5",
            "--min-free-vram-gib",
            "20",
        ],
    )

    with pytest.raises(SystemExit, match="no visible GPU has the required 20.0 GiB free"):
        hardware.main()
