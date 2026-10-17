from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path

import pytest

import oci.evaluation.benchmark_driver as driver
from oci.evaluation.benchmark_driver import DriverError, GenerationJob

SOURCE = driver.DEFAULT_SOURCE


def _environ(**values: str) -> dict[str, str]:
    environ = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("STAGE1_", "STAGE2_", "OCI_", "GENERATOR_", "CUDA_"))
        and key not in {"PRESET", "GPU_COUNT", "PHYSICAL_GPUS", "MIN_FREE_GPU_GB"}
    }
    environ.update(values)
    return environ


def _complete_run(run_dir: Path) -> None:
    stage2 = run_dir / "stage2"
    stage2.mkdir(parents=True)
    (stage2 / "complete.json").write_text(
        json.dumps({"status": "complete", "phase": "causal_estimation"}), encoding="utf-8"
    )
    for name in ("cross_fitted_predictions.csv", "causal_estimate.json",
                 "heterogeneity_diagnostics.json"):
        (stage2 / name).write_text("{}", encoding="utf-8")


def _args(machine: str, out: Path, *options: str):
    return driver.build_parser().parse_args([machine, str(out), *options])


def test_preset_environment_reads_presets_and_keeps_explicit_values():
    gb300 = driver.preset_environment("dgx_station_gb300", _environ())
    assert gb300["PHYSICAL_GPUS"] == "1"
    assert gb300["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert gb300["STAGE2_MODEL"] == "nvidia/Gemma-4-26B-A4B-NVFP4"
    rtx = driver.preset_environment("rtxpro6000x8", _environ(STAGE2_MODEL="custom"))
    assert rtx["STAGE2_MODEL"] == "custom" and rtx["GPU_COUNT"] == "8"
    assert "dgx_station_gb300" in driver.machines()
    with pytest.raises(DriverError, match="unknown machine"):
        driver.preset_environment("a100x4", _environ())


def test_plan_uses_the_source_when_it_covers_the_selected_cells(tmp_path):
    plan = driver.build_plan(machine="rtxpro6000x8", bench=tmp_path, scenarios="S0,S1",
                             sizes="1000", seeds="101")

    assert plan.jobs == ()
    assert plan.reequation_source == SOURCE
    assert plan.source_record == {"source": str(SOURCE), "expanded_to": None}
    assert [(cell.id, variant) for cell, variant in plan.runs] == [
        ("S0/n1000/seed101", "default"), ("S1/n1000/seed101", "default"),
    ]


def test_plan_expands_the_source_and_generates_scenario_corpora(tmp_path):
    plan = driver.build_plan(machine="rtxpro6000x8", bench=tmp_path, scenarios="S0,S6",
                             sizes="1000,5000", seeds="101")

    expansion, corpus = plan.jobs
    assert expansion.name == "source_expansion" and expansion.patients == 5000
    assert expansion.dgp_metadata == SOURCE / "metadata.json"
    assert plan.reequation_source == expansion.output_dir == (
        tmp_path / "corpora" / f"{SOURCE.name}_n5000"
    )
    assert corpus.name == "S6" and corpus.patients == 5000
    assert corpus.output_dir == tmp_path / "S6" / "generation" / "corpus"
    assert corpus.settings["post_treatment_notes"] == {"rate": 1.0}
    # Smallest cells first; S6 has two run variants per cell.
    assert [(cell.id, variant) for cell, variant in plan.runs][:3] == [
        ("S0/n1000/seed101", "default"),
        ("S6/n1000/seed101", "dated_notes"),
        ("S6/n1000/seed101", "text_column"),
    ]


def test_plan_keeps_one_reequation_source_per_benchmark_directory(tmp_path):
    with pytest.raises(DriverError, match="smaller than the largest"):
        driver.build_plan(machine="external", bench=tmp_path, scenarios="S0",
                          sizes="5000", seeds="101", source_size=2000)
    state = tmp_path / "driver" / "state.json"
    state.parent.mkdir(parents=True)
    state.write_text(json.dumps({"reequation_source": {
        "source": str(SOURCE), "expanded_to": None,
    }}), encoding="utf-8")
    driver.build_plan(machine="external", bench=tmp_path, scenarios="S0", sizes="1000",
                      seeds="101")
    with pytest.raises(DriverError, match="--source-size"):
        driver.build_plan(machine="external", bench=tmp_path, scenarios="S0", sizes="5000",
                          seeds="101")
    # A generated-corpus scenario alone does not touch the source.
    driver.build_plan(machine="external", bench=tmp_path, scenarios="S5", seeds="101")


def test_generator_command_routes_requests_to_every_replica(tmp_path):
    expansion = GenerationJob(
        name="source_expansion", output_dir=tmp_path / "expanded",
        settings=json.loads((SOURCE / "generation_config.json").read_text(encoding="utf-8")),
        patients=10_000, dgp_metadata=SOURCE / "metadata.json",
    )
    command = driver.generator_command(
        expansion, bench=tmp_path, endpoints=["http://a/v1", "http://b/v1"],
        model="openai/gpt-oss-120b", workers=64,
    )
    config_path = Path(command[command.index("--config") + 1])
    settings = json.loads(config_path.read_text(encoding="utf-8"))
    assert command[1:3] == ["-m", "synthetic_data.cli"]
    assert command[command.index("--dgp-metadata") + 1] == str(SOURCE / "metadata.json")
    assert command[command.index("--num-workers") + 1] == "64"
    assert command[command.index("--output-dir") + 1] == str(tmp_path / "expanded")
    assert "--api-key" not in command
    assert settings["dataset_size"] == 10_000
    assert settings["llm"]["api_base_urls"] == ["http://a/v1", "http://b/v1"]
    assert settings["llm"]["model_name"] == "openai/gpt-oss-120b"
    assert settings["llm"]["max_tokens"] == 50_000 and settings["llm"]["api_key"] == ""

    corpus = GenerationJob(name="S5", output_dir=tmp_path / "S5", settings={"seed": 42},
                           patients=5000)
    command = driver.generator_command(
        corpus, bench=tmp_path, endpoints=["http://a/v1"], model="m", workers=8,
        api_key="secret",
    )
    settings = json.loads(Path(command[command.index("--config") + 1]).read_text())
    assert "--dgp-metadata" not in command
    assert command[command.index("--api-key") + 1] == "secret"
    assert "api_base_urls" not in settings["llm"] and settings["llm"]["api_key"] == ""
    assert settings["llm"]["max_tokens"] == 50_000


def test_managed_generation_pool_uses_the_preset_gpus_and_restores_cuda(tmp_path, monkeypatch):
    plan = driver.build_plan(machine="dgx_station_gb300", bench=tmp_path, scenarios="S5",
                             seeds="101")
    env = driver.preset_environment("dgx_station_gb300", _environ())
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    seen = {}

    def detect(cuda_env):
        seen["detect"] = {k: cuda_env.get(k) for k in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")}
        return "cuda:0"

    @contextlib.contextmanager
    def launch(*, config, model, api_key, output_dir):
        seen.update(config=config, model=model, api_key=api_key, output_dir=output_dir,
                    visible=os.environ.get("CUDA_VISIBLE_DEVICES"))
        yield ("http://127.0.0.1:8210/v1",)

    with driver.generation_endpoints(plan, env, detect=detect, launch=launch) as (endpoints, replicas):
        assert endpoints == ("http://127.0.0.1:8210/v1",) and replicas == 1
    assert seen["detect"] == {"CUDA_VISIBLE_DEVICES": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    assert seen["visible"] == "1"
    assert seen["model"] == "openai/gpt-oss-120b" and seen["api_key"] == "EMPTY"
    config = seen["config"]
    assert config.gpus == ("cuda:0",) and config.server_count == 1
    assert config.base_port == 8210 and config.internal_port_base == 40_000
    assert list(config.extra_args) == [
        "--gpu-memory-utilization", "0.90", "--max-model-len", "131072",
    ]
    assert seen["output_dir"] == tmp_path / "driver" / "generation_servers"
    assert "CUDA_VISIBLE_DEVICES" not in os.environ and "CUDA_DEVICE_ORDER" not in os.environ


def test_external_generator_endpoints_skip_the_pool(tmp_path):
    plan = driver.build_plan(machine="rtxpro6000x8", bench=tmp_path, scenarios="S5", seeds="101")
    env = driver.preset_environment(
        "rtxpro6000x8", _environ(GENERATOR_ENDPOINT="http://x/v1, http://y/v1")
    )

    def fail(*_args, **_kwargs):
        raise AssertionError("no pool or GPU detection for external endpoints")

    with driver.generation_endpoints(plan, env, detect=fail, launch=fail) as (endpoints, replicas):
        assert endpoints == ("http://x/v1", "http://y/v1") and replicas == 2


def test_generation_fails_closed_when_the_corpus_is_incomplete(tmp_path):
    plan = driver.build_plan(machine="external", bench=tmp_path, scenarios="S5", seeds="101")
    commands = []

    def runner(command, log, env):
        commands.append(command)
        return 0

    @contextlib.contextmanager
    def endpoints(_plan, _env):
        yield ("http://x/v1",), 1

    with pytest.raises(DriverError, match="did not finish"):
        driver.run_generation(plan, {}, runner=runner, endpoints_context=endpoints)
    assert commands[0][commands[0].index("--num-workers") + 1] == "32"


def test_runs_skip_finished_cells_and_continue_after_a_failure(tmp_path):
    plan = driver.build_plan(machine="h100x8", bench=tmp_path, scenarios="S0,S1,S2",
                             sizes="1000", seeds="101")
    first, second, third = (cell for cell, _variant in plan.runs)
    _complete_run(first.run_dir(tmp_path, "default"))
    calls = []

    def runner(command, log, env):
        calls.append((command, env["PRESET"], env["OCI_PYTHON"]))
        if str(third.config_path(tmp_path, "default")) == command[-1]:
            _complete_run(third.run_dir(tmp_path, "default"))
            return 0
        return 3

    results = driver.run_cells(plan, {"PATH": os.environ["PATH"]}, runner=runner)
    assert [item["result"] for item in results] == ["complete", "failed", "complete"]
    assert [command[-1] for command, _, _ in calls] == [
        str(second.config_path(tmp_path, "default")), str(third.config_path(tmp_path, "default")),
    ]
    assert all(command[:2] == ["bash", str(driver.LAUNCHER)] for command, _, _ in calls)
    assert {preset for _, preset, _ in calls} == {"h100x8"}

    calls.clear()
    stopped = driver.run_cells(plan, {}, runner=lambda *a: 1, stop_on_failure=True)
    assert [item["result"] for item in stopped] == ["complete", "failed"]


def test_driver_runs_every_step_without_llm_text(tmp_path, capsys):
    bench = tmp_path / "bench"
    launched = []

    def runner(command, log, env):
        launched.append(command[-1])
        _complete_run(Path(json.loads(Path(command[-1]).read_text())["output_dir"]))
        return 0

    def endpoints(*_args, **_kwargs):
        raise AssertionError("no corpus is needed")

    args = _args("rtxpro6000x8", bench, "--scenarios", "S0", "--sizes", "1000",
                 "--seeds", "101", "--through", "runs")
    assert driver.run(args, environ=_environ(), runner=runner, endpoints_context=endpoints) == 0
    cell = bench / "S0" / "n1000" / "seed101"
    config = json.loads((cell / "runs" / "default" / "config.json").read_text())
    assert launched == [str(cell / "runs" / "default" / "config.json")]
    assert config["dataset"] == str(cell / "data" / "dataset.parquet")
    assert config["science"]["seed"] == 101
    assert config["stage2"]["consolidation_max_rounds"] == 55
    assert "endpoint" not in config["stage2"] and "vllm" not in config["stage2"]
    invocation = json.loads((bench / "driver" / "invocations.jsonl").read_text().splitlines()[-1])
    assert invocation["machine"] == "rtxpro6000x8"
    assert invocation["stage2_model"] == "nvidia/Gemma-4-26B-A4B-NVFP4"

    # A rerun finds the run finished and launches nothing.
    launched.clear()
    assert driver.run(args, environ=_environ(), runner=runner, endpoints_context=endpoints) == 0
    assert launched == []
    assert "S0/n1000/seed101 default: complete" in capsys.readouterr().out


def test_confirmatory_configurations_are_written_once_then_resumed(tmp_path):
    bench = tmp_path / "bench"
    ledger = tmp_path / "ledger.jsonl"
    args = _args("external", bench, "--scenarios", "S0", "--sizes", "1000", "--seeds", "7001",
                 "--confirmatory", "--ledger", str(ledger), "--through", "configs")
    assert driver.run(args, environ=_environ()) == 0
    assert len(ledger.read_text().splitlines()) == 1
    # Resuming does not write a second ledger entry or trip the one-use rule.
    assert driver.run(args, environ=_environ()) == 0
    assert len(ledger.read_text().splitlines()) == 1


def test_plan_only_writes_nothing(tmp_path):
    bench = tmp_path / "bench"
    assert driver.main(["dgx_station_gb300", str(bench), "--plan"]) == 0
    assert not bench.exists()
    assert driver.main(["rtxpro6000x8", str(bench), "--seeds", "7001", "--plan"]) == 2
