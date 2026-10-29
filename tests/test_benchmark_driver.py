from __future__ import annotations

import json
import os
from pathlib import Path


import oci.evaluation.benchmark_driver as driver


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


def test_plan_only_writes_nothing(tmp_path):
    bench = tmp_path / "bench"
    assert driver.main(["dgx_station_gb300", str(bench), "--plan"]) == 0
    assert not bench.exists()
    assert driver.main(["rtxpro6000x8", str(bench), "--seeds", "7001", "--plan"]) == 2
