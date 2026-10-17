"""End-to-end benchmark driver: corpora, cohorts, configurations, runs, aggregate.

``run_benchmark.sh MACHINE BENCH`` (this module's :func:`main`) runs every step
of the benchmark harness (:mod:`oci.evaluation.benchmark`) on one machine, in
order, and resumes from where it stopped when rerun:

1. ``generate``: LLM text that the selected cells need. When a selected
   re-equation cell (S0 to S4) is larger than the re-equation source, the
   source is expanded with its saved DGP; each selected generated-corpus
   scenario (S5, S6) gets its corpus. The generator runs on a vLLM pool that
   the driver starts on the machine's GPUs and stops before any run, or on
   ``GENERATOR_ENDPOINT``.
2. ``materialize``: one cohort per scenario, size, and seed.
3. ``configs``: one workflow configuration per cell from the base
   configuration (``benchmarks/base_config.json`` by default).
4. ``runs``: each unfinished run, one at a time, through
   ``run_synthetic.sh CONFIG`` with ``PRESET=MACHINE``, which adds the
   machine's serving settings and the detected devices and workers.
5. ``aggregate``: finished runs versus oracle truth.

``MACHINE`` names a preset in ``scripts/presets/``. The driver reads the preset
(and the environment, whose values win) for the generator's GPUs and for its
report; the runs read it themselves. With ``--llm-url``, running vLLM servers
replace the preset's managed ones for both Stage 2 roles and the generator,
and the preset contributes only its hardware settings (GPU selection, CPU
workers, and timeouts). Oracle columns are read only by the harness's
aggregate step, as when it is run by hand.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shlex
import subprocess
import sys
import urllib.request
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .benchmark import (
    REPOSITORY_ROOT,
    BENCHMARK_DIR,
    Cell,
    Scenario,
    SeedGuardError,
    _parse_sizes,
    _parquet_rows,
    aggregate,
    expand_cells,
    git_state,
    load_registry,
    load_seed_registry,
    materialize,
    resolve_seeds,
    run_status,
    write_configs,
)

DRIVER_SCHEMA_VERSION = "oci_benchmark_driver_v1"
PRESET_DIR = REPOSITORY_ROOT / "scripts" / "presets"
LAUNCHER = REPOSITORY_ROOT / "run_synthetic.sh"
DETECT_HARDWARE = REPOSITORY_ROOT / "scripts" / "detect_all_evidence_hardware.py"
DEFAULT_BASE = BENCHMARK_DIR / "base_config.json"
DEFAULT_SOURCE = (
    REPOSITORY_ROOT / "synthetic_data" / "example_synthetic_datasets"
    / "five_confounders_five_effect_modifiers_nsclc_with_structured"
)
STEPS = ("generate", "materialize", "configs", "runs", "aggregate")
# The model that wrote the bundled cohorts and the generator's own default.
DEFAULT_GENERATOR_MODEL = "openai/gpt-oss-120b"
DEFAULT_GENERATOR_ENDPOINT = "http://127.0.0.1:8000/v1"
DEFAULT_GENERATOR_WORKERS_PER_SERVER = 32
# The generator command line's output ceiling; its configuration files
# otherwise fall back to LLMConfig's smaller 12,000.
GENERATOR_MAX_TOKENS = 50_000
# Seconds per generator request. LLMConfig's 300 fails requests that wait in
# a busy server's queue; this matches Stage 2's attempt timeout. The timeout
# only routes requests, so an interrupted generation still resumes.
GENERATOR_REQUEST_TIMEOUT = 1800.0
# The generator shares one OpenAI client per server, whose connection pool
# holds 1,000 connections; more patient threads would wait for a connection.
GENERATOR_MAX_WORKERS_PER_SERVER = 1000
GENERATOR_VLLM_DEFAULTS = {
    "GENERATOR_VLLM_GPUS_PER_SERVER": "1",
    "GENERATOR_VLLM_BASE_PORT": "8210",
    "GENERATOR_VLLM_INTERNAL_PORT_BASE": "40000",
    "GENERATOR_VLLM_EXTRA_ARGS_JSON": json.dumps(
        ["--gpu-memory-utilization", "0.90", "--max-model-len", "131072"]
    ),
}

# Preset settings kept when --llm-url replaces the preset's servers: Stage 1's
# GPU selection and CPU budget, final-modeling CPU workers, and timeouts.
HARDWARE_VARIABLES = (
    "PHYSICAL_GPUS", "GPU_COUNT", "MIN_FREE_GPU_GB", "MIN_TOTAL_GPU_GB", "CUDA_DEVICE_ORDER",
    "STAGE1_WORKERS", "STAGE2_MODELING_WORKERS", "STAGE2_REQUEST_TIMEOUT",
    "STAGE2_REQUEST_ATTEMPT_TIMEOUT",
)
# Serving variables that --llm-url sets; an explicit value would contradict it.
LLM_URL_CONFLICTS = (
    "STAGE2_ENDPOINT", "STAGE2_MODEL", "STAGE2_EXTRACTION_ENDPOINT", "STAGE2_EXTRACTION_ENDPOINTS",
    "STAGE2_EXTRACTION_MODEL",
)
CONTEXT_MARGIN_TOKENS = 4096
# Room for the generator's prompts when a server's window caps its output.
GENERATOR_PROMPT_RESERVE_TOKENS = 16_384

Runner = Callable[[Sequence[str], Path, Mapping[str, str]], int]


class DriverError(RuntimeError):
    """A step cannot proceed; the message says why and what to do."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def machines() -> list[str]:
    return sorted(path.stem for path in PRESET_DIR.glob("*.env"))


def preset_environment(machine: str, environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment after sourcing the machine's preset (set values win)."""

    path = PRESET_DIR / f"{machine}.env"
    if not path.is_file():
        raise DriverError(f"unknown machine {machine!r}; choose one of: {', '.join(machines())}")
    completed = subprocess.run(
        ["bash", "-c", 'set -a; source "$1" >/dev/null; env -0', "_", str(path)],
        env=dict(os.environ if environ is None else environ),
        capture_output=True,
        check=True,
    )
    return dict(
        item.split("=", 1)
        for item in completed.stdout.decode("utf-8", "replace").split("\0")
        if "=" in item
    )


def run_logged(command: Sequence[str], log_path: Path, env: Mapping[str, str]) -> int:
    """Run a command from the repository root, copying its output to the console and a log."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n# {_now()} {shlex.join(map(str, command))}\n")
        log.flush()
        process = subprocess.Popen(
            [str(part) for part in command], cwd=REPOSITORY_ROOT, env=dict(env),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                log.write(line)
                sys.stdout.write(line)
            return process.wait()
        except KeyboardInterrupt:
            # The child received the same interrupt and stops its own servers.
            process.wait()
            raise


# ---------------------------------------------------------------------------
# running vLLM servers (--llm-url)


@dataclass(frozen=True)
class LLMServers:
    """Running OpenAI-compatible servers that serve one model for every LLM role."""

    urls: tuple[str, ...]
    model: str
    context_window: int | None
    concurrency: int

    @property
    def interpretation_workers(self) -> int:
        return max(1, self.concurrency // 4)

    def extraction_endpoints(self) -> list[dict[str, Any]]:
        """Per-server extraction limits; the first server also takes interpretation."""

        return [
            {"endpoint": url,
             "max_concurrency": self.concurrency - (self.interpretation_workers if index == 0 else 0)}
            for index, url in enumerate(self.urls)
        ]


def _fetch_models(url: str) -> Any:
    with urllib.request.urlopen(f"{url.rstrip('/')}/models", timeout=15) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def probe_llm_servers(
    urls: str | Sequence[str],
    *,
    concurrency: int,
    fetch: Callable[[str], Any] = _fetch_models,
) -> LLMServers:
    """Read each server's model and context window from its ``/models`` list."""

    if isinstance(urls, str):
        urls = [part.strip().rstrip("/") for part in urls.split(",") if part.strip()]
    if not urls:
        raise DriverError("--llm-url needs at least one URL")
    if int(concurrency) < 2:
        raise DriverError("--llm-concurrency must be at least 2 (interpretation and extraction)")
    models: dict[str, str] = {}
    windows: list[int] = []
    for url in urls:
        try:
            payload = fetch(url)
        except Exception as error:  # noqa: BLE001 - any failure means the server is unusable
            raise DriverError(f"cannot read {url}/models: {error}") from error
        data = payload.get("data") if isinstance(payload, Mapping) else None
        if not isinstance(data, list) or len(data) != 1:
            raise DriverError(
                f"{url} serves {len(data) if isinstance(data, list) else 'no'} models; "
                "--llm-url needs servers that each serve exactly one model"
            )
        models[url] = str(data[0].get("id") or "")
        window = data[0].get("max_model_len")
        if isinstance(window, int) and not isinstance(window, bool) and window > 0:
            windows.append(window)
    if len(set(models.values())) != 1 or not next(iter(models.values())):
        raise DriverError(f"the --llm-url servers must serve one model; they serve {models}")
    return LLMServers(
        urls=tuple(urls),
        model=next(iter(models.values())),
        context_window=min(windows) if windows else None,
        concurrency=int(concurrency),
    )


def external_run_environment(
    environ: Mapping[str, str],
    machine_env: Mapping[str, str],
    servers: LLMServers,
) -> dict[str, str]:
    """The runs' environment: the external preset on these servers, the machine's hardware."""

    conflicts = sorted(
        name for name in environ
        if name in LLM_URL_CONFLICTS or name.startswith(("STAGE2_VLLM_", "STAGE2_EXTRACTION_VLLM_"))
    )
    if conflicts:
        raise DriverError(f"--llm-url sets the Stage 2 servers; unset {', '.join(conflicts)}")
    derived = {
        "STAGE2_WORKERS": str(servers.interpretation_workers),
        **({"STAGE2_EXTRACTION_CONTEXT_WINDOW_TOKENS": str(servers.context_window),
            "STAGE2_EXTRACTION_CONTEXT_MARGIN_TOKENS": str(CONTEXT_MARGIN_TOKENS)}
           if servers.context_window else {}),
        **{name: machine_env[name] for name in HARDWARE_VARIABLES if machine_env.get(name)},
    }
    return {
        **derived,
        # Explicit settings (workers, windows, timeouts) win over derived ones.
        **environ,
        "PRESET": "external",
        "STAGE2_ENDPOINT": servers.urls[0],
        "STAGE2_MODEL": servers.model,
        "STAGE2_EXTRACTION_ENDPOINTS": json.dumps(servers.extraction_endpoints()),
        "STAGE2_EXTRACTION_MODEL": servers.model,
    }


def generator_environment(machine_env: Mapping[str, str], servers: LLMServers) -> dict[str, str]:
    """The generator uses the servers too unless GENERATOR_ENDPOINT names others."""

    env = dict(machine_env)
    if not env.get("GENERATOR_ENDPOINT"):
        env["GENERATOR_ENDPOINT"] = ",".join(servers.urls)
        env.setdefault("GENERATOR_MODEL", servers.model)
        env.setdefault("GENERATOR_WORKERS", str(
            min(servers.concurrency, GENERATOR_MAX_WORKERS_PER_SERVER) * len(servers.urls)
        ))
    return env


def generator_max_tokens(servers: LLMServers | None, env: Mapping[str, str]) -> int | None:
    """The generator's output ceiling, capped to fit the servers' window."""

    if servers is None or servers.context_window is None:
        return None
    if env.get("GENERATOR_ENDPOINT") != ",".join(servers.urls):
        return None
    return max(4096, min(GENERATOR_MAX_TOKENS,
                         servers.context_window - GENERATOR_PROMPT_RESERVE_TOKENS))


# ---------------------------------------------------------------------------
# plan


@dataclass(frozen=True)
class GenerationJob:
    """One generator run: a saved-DGP expansion of the source, or a scenario corpus."""

    name: str
    output_dir: Path
    settings: Mapping[str, Any]
    patients: int
    dgp_metadata: Path | None = None

    def complete(self) -> bool:
        if not (self.output_dir / "dataset.parquet").is_file():
            return False
        metadata_path = self.output_dir / "metadata.json"
        if not metadata_path.is_file():
            return False
        if self.dgp_metadata is None:
            return True
        try:
            return _read_json(metadata_path).get("generation_status") == "complete"
        except (OSError, ValueError):
            return False


@dataclass(frozen=True)
class Plan:
    machine: str
    bench: Path
    scenarios: tuple[Scenario, ...]
    cells: tuple[Cell, ...]
    jobs: tuple[GenerationJob, ...]
    reequation_source: Path | None
    source_record: Mapping[str, Any] | None
    corpora: Mapping[str, Path]

    @property
    def runs(self) -> list[tuple[Cell, str]]:
        order = {scenario.id: index for index, scenario in enumerate(self.scenarios)}
        pairs = [(cell, variant.name) for cell in self.cells for variant in cell.scenario.variants]
        return sorted(pairs, key=lambda pair: (
            pair[0].size, order[pair[0].scenario.id], pair[0].seed, pair[1],
        ))


def _state_path(bench: Path) -> Path:
    return bench / "driver" / "state.json"


def build_plan(
    *,
    machine: str,
    bench: Path,
    scenarios: str | None = None,
    sizes: str | None = None,
    seeds: str | None = None,
    confirmatory: bool = False,
    source: Path | None = None,
    source_size: int | None = None,
    registry_path: Path | None = None,
    seeds_path: Path | None = None,
) -> Plan:
    bench = Path(bench).expanduser().resolve()
    registry = load_registry(registry_path)
    selected = registry.select(scenarios)
    seed_pairs = resolve_seeds(load_seed_registry(seeds_path), seeds, confirmatory=confirmatory)
    cells = expand_cells(selected, seed_pairs, _parse_sizes(sizes))
    if not cells:
        raise DriverError("the selection contains no cells; check --scenarios and --sizes")
    jobs: list[GenerationJob] = []
    corpora: dict[str, Path] = {}

    reequation_cells = [cell for cell in cells if not cell.scenario.needs_generated_corpus]
    reequation_source: Path | None = None
    record: dict[str, Any] | None = None
    if reequation_cells:
        origin = Path(source or DEFAULT_SOURCE).expanduser().resolve()
        if not (origin / "dataset.parquet").is_file():
            raise DriverError(f"re-equation source {origin} has no dataset.parquet")
        available = _parquet_rows(origin / "dataset.parquet")
        needed = max(cell.size for cell in reequation_cells)
        if source_size is not None and source_size < needed:
            raise DriverError(
                f"--source-size {source_size} is smaller than the largest selected "
                f"re-equation cell ({needed} patients)"
            )
        target = max(needed, source_size or 0)
        if target <= available:
            reequation_source = origin
            record = {"source": str(origin), "expanded_to": None}
        else:
            missing = [name for name in ("generation_config.json", "metadata.json")
                       if not (origin / name).is_file()]
            if missing:
                raise DriverError(
                    f"cells of up to {target} patients need the {available}-patient source "
                    f"{origin} expanded with its saved DGP, but it lacks {missing}"
                )
            reequation_source = bench / "corpora" / f"{origin.name}_n{target}"
            jobs.append(GenerationJob(
                name="source_expansion",
                output_dir=reequation_source,
                settings=_read_json(origin / "generation_config.json"),
                patients=target,
                dgp_metadata=origin / "metadata.json",
            ))
            record = {"source": str(origin), "expanded_to": target}
        state_path = _state_path(bench)
        recorded = (_read_json(state_path).get("reequation_source")
                    if state_path.is_file() else None)
        if recorded is not None and recorded != record:
            raise DriverError(
                f"{bench} uses the re-equation source {json.dumps(recorded)}, but this "
                f"selection needs {json.dumps(record)}. Every S0-S4 cell of a benchmark "
                "directory comes from one source: use a new BENCH directory, or plan the "
                "largest size from the start with --source-size"
            )

    generated_ids = {cell.scenario.id for cell in cells if cell.scenario.needs_generated_corpus}
    for scenario in (s for s in selected if s.id in generated_ids):
        if not scenario.generation:
            raise DriverError(f"scenario {scenario.id} has no generation settings in the registry")
        corpus = bench / scenario.id / "generation" / "corpus"
        corpora[scenario.id] = corpus
        jobs.append(GenerationJob(
            name=scenario.id,
            output_dir=corpus,
            settings=dict(scenario.generation),
            patients=int(scenario.generation.get("dataset_size") or scenario.minimum_source_patients),
        ))
    return Plan(
        machine=machine,
        bench=bench,
        scenarios=tuple(selected),
        cells=tuple(cells),
        jobs=tuple(jobs),
        reequation_source=reequation_source,
        source_record=record,
        corpora=corpora,
    )


def describe(plan: Plan, env: Mapping[str, str], servers: LLMServers | None = None) -> str:
    managed = not _external_generator(plan.machine, env)
    lines = [
        f"Machine:      {plan.machine} (scripts/presets/{plan.machine}.env)",
        f"Benchmark:    {plan.bench}",
    ]
    if servers is not None:
        window = (f"{servers.context_window}-token window" if servers.context_window
                  else "window not reported")
        lines.append(
            f"Stage 2:      {servers.model} at {', '.join(servers.urls)} ({window}; "
            f"{servers.interpretation_workers} interpretation and "
            f"{sum(e['max_concurrency'] for e in servers.extraction_endpoints())} extraction "
            "requests at a time), in place of the preset's servers"
        )
    else:
        primary = env.get("STAGE2_MODEL") or "(not set)"
        extraction = env.get("STAGE2_EXTRACTION_MODEL") or "(not set)"
        lines.append(f"Stage 2:      {primary} (interpretation), {extraction} (extraction)")
    if plan.jobs:
        where = ("a managed vLLM pool on "
                 + (env.get("GENERATOR_VLLM_GPUS") or "the detected GPUs")
                 if managed else ", ".join(_generator_endpoints(plan.machine, env)))
        lines.append(f"Generator:    {_generator_model(env)} on {where}")
    if plan.reequation_source is not None:
        lines.append(f"Re-equation source for S0-S4: {plan.reequation_source}")
    for job in plan.jobs:
        state = "complete" if job.complete() else "pending (LLM generation)"
        lines.append(f"Corpus {job.name}: {job.patients} patients at {job.output_dir}: {state}")
    materialized = sum((cell.directory(plan.bench) / "materialize.json").is_file()
                       for cell in plan.cells)
    lines.append(f"Cells:        {len(plan.cells)} ({materialized} materialized)")
    runs = plan.runs
    finished = sum(run_status(cell.run_dir(plan.bench, variant))[0] == "complete"
                   for cell, variant in runs)
    lines.append(f"Runs:         {len(runs)} ({finished} complete), smallest cells first")
    for cell, variant in runs:
        status, detail = run_status(cell.run_dir(plan.bench, variant))
        lines.append(f"  {cell.id} {variant}: {detail if status != 'complete' else 'complete'}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# generate


def _generator_model(env: Mapping[str, str]) -> str:
    return env.get("GENERATOR_MODEL") or DEFAULT_GENERATOR_MODEL


def _external_generator(machine: str, env: Mapping[str, str]) -> bool:
    return bool(env.get("GENERATOR_ENDPOINT")) or machine == "external"


def _generator_endpoints(machine: str, env: Mapping[str, str]) -> list[str]:
    text = env.get("GENERATOR_ENDPOINT") or DEFAULT_GENERATOR_ENDPOINT
    return [part.strip() for part in text.split(",") if part.strip()]


def detect_devices(env: Mapping[str, str]) -> str:
    """Logical CUDA devices that run_synthetic.sh would select under this environment."""

    command = [
        sys.executable, str(DETECT_HARDWARE), "--outer-folds", "5", "--inner-folds", "5",
        "--gpu-count", env.get("GPU_COUNT") or "auto",
        "--min-free-vram-gib", env.get("MIN_FREE_GPU_GB") or "0",
        "--min-total-vram-gib", env.get("MIN_TOTAL_GPU_GB") or "0",
    ]
    completed = subprocess.run(command, env=dict(env), capture_output=True, text=True)
    if completed.returncode != 0:
        raise DriverError(f"GPU detection failed: {(completed.stderr or completed.stdout).strip()}")
    fields = completed.stdout.strip().split("\t")
    if len(fields) < 6 or not fields[1]:
        raise DriverError(f"GPU detection returned an incomplete result: {completed.stdout!r}")
    print(f"Generator GPUs: {fields[5]}")
    return fields[1]


def _cuda_environment(env: Mapping[str, str]) -> dict[str, str]:
    """The driver's environment with the preset's GPU mapping, as run_synthetic.sh applies it."""

    mapped = dict(env)
    if env.get("PHYSICAL_GPUS"):
        mapped["CUDA_VISIBLE_DEVICES"] = env["PHYSICAL_GPUS"]
    return mapped


@contextlib.contextmanager
def generation_endpoints(
    plan: Plan,
    env: Mapping[str, str],
    *,
    detect: Callable[[Mapping[str, str]], str] = detect_devices,
    launch: Callable[..., Any] | None = None,
) -> Iterator[tuple[tuple[str, ...], int]]:
    """Endpoints for the generator and the number of replicas behind them."""

    if _external_generator(plan.machine, env):
        endpoints = tuple(_generator_endpoints(plan.machine, env))
        yield endpoints, len(endpoints)
        return
    from oci.inference.vllm_server_pool import (
        launch_managed_vllm_servers,
        managed_vllm_config_from_mapping,
    )

    settings = {**GENERATOR_VLLM_DEFAULTS, **{k: v for k, v in env.items() if v}}
    cuda_env = _cuda_environment(env)
    saved = {name: os.environ.get(name) for name in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")}
    try:
        # The pool maps logical GPUs through the parent's CUDA_VISIBLE_DEVICES.
        for name in saved:
            if cuda_env.get(name):
                os.environ[name] = cuda_env[name]
        gpus = settings.get("GENERATOR_VLLM_GPUS") or detect(cuda_env)
        config = managed_vllm_config_from_mapping(
            {
                "gpus": gpus,
                "gpus_per_server": int(settings["GENERATOR_VLLM_GPUS_PER_SERVER"]),
                "base_port": int(settings["GENERATOR_VLLM_BASE_PORT"]),
                "internal_port_base": int(settings["GENERATOR_VLLM_INTERNAL_PORT_BASE"]),
                "download_dir": settings.get("STAGE2_VLLM_DOWNLOAD_DIR", ""),
                "extra_args": json.loads(settings["GENERATOR_VLLM_EXTRA_ARGS_JSON"]),
            },
            model=_generator_model(env),
            section="GENERATOR_VLLM",
        )
        assert config is not None
        print(f"Starting {config.server_count} generator vLLM server(s) for "
              f"{_generator_model(env)} on {', '.join(config.gpus)}")
        with (launch or launch_managed_vllm_servers)(
            config=config, model=_generator_model(env), api_key="EMPTY",
            output_dir=plan.bench / "driver" / "generation_servers",
        ) as endpoints:
            yield tuple(endpoints), config.server_count
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def generator_command(
    job: GenerationJob,
    *,
    bench: Path,
    endpoints: Sequence[str],
    model: str,
    workers: int,
    api_key: str = "",
    max_tokens: int | None = None,
) -> list[str]:
    """Write the job's generator configuration (routed to the endpoints) and return its command."""

    settings = json.loads(json.dumps(dict(job.settings)))
    llm = dict(settings.get("llm") or {})
    llm.update(model_name=model, api_base_url=endpoints[0], api_key="")
    llm.setdefault("max_tokens", GENERATOR_MAX_TOKENS)
    llm["timeout"] = max(float(llm.get("timeout") or 0.0), GENERATOR_REQUEST_TIMEOUT)
    if max_tokens is not None:
        llm["max_tokens"] = min(int(llm["max_tokens"]), int(max_tokens))
    if len(endpoints) > 1:
        llm["api_base_urls"] = list(endpoints)
    else:
        llm.pop("api_base_urls", None)
    settings["llm"] = llm
    settings["dataset_size"] = int(job.patients)
    config_path = bench / "driver" / "generation" / f"{job.name}.generator_config.json"
    _write_json(config_path, settings)
    command = [
        sys.executable, "-m", "synthetic_data.cli", "--config", str(config_path),
        "--num-workers", str(int(workers)), "--output-dir", str(job.output_dir),
    ]
    if job.dgp_metadata is not None:
        command += ["--dgp-metadata", str(job.dgp_metadata)]
    if api_key:
        command += ["--api-key", api_key]
    return command


def run_generation(
    plan: Plan,
    env: Mapping[str, str],
    *,
    runner: Runner = run_logged,
    endpoints_context: Callable[..., Any] = generation_endpoints,
    max_tokens: int | None = None,
) -> None:
    pending = [job for job in plan.jobs if not job.complete()]
    if not pending:
        return
    model = _generator_model(env)
    with endpoints_context(plan, env) as (endpoints, replicas):
        workers = int(env.get("GENERATOR_WORKERS")
                      or DEFAULT_GENERATOR_WORKERS_PER_SERVER * max(1, replicas))
        for job in pending:
            log = plan.bench / "driver" / "logs" / f"generate_{job.name}.log"
            print(f"\n== generate {job.name}: {job.patients} patients with {model} "
                  f"({workers} workers) -> {job.output_dir}\n   log: {log}")
            command = generator_command(
                job, bench=plan.bench, endpoints=endpoints, model=model, workers=workers,
                api_key=env.get("GENERATOR_API_KEY", ""), max_tokens=max_tokens,
            )
            code = runner(command, log, env)
            if code != 0 or not job.complete():
                raise DriverError(
                    f"generation of {job.name} did not finish (exit code {code}); see {log}. "
                    "Rerun the same command to resume it."
                )


# ---------------------------------------------------------------------------
# materialize, configs, runs


def run_materialize(plan: Plan, *, confirmatory: bool, seeds: str | None, sizes: str | None,
                    registry_path: Path | None, seeds_path: Path | None) -> None:
    shared = {"out": plan.bench, "sizes": sizes, "seeds": seeds, "confirmatory": confirmatory,
              "registry_path": registry_path, "seeds_path": seeds_path}
    reequation = list(dict.fromkeys(
        cell.scenario.id for cell in plan.cells if not cell.scenario.needs_generated_corpus
    ))
    if reequation:
        materialize(scenarios=",".join(reequation), source=plan.reequation_source, **shared)
        if plan.source_record is not None:
            _write_json(_state_path(plan.bench), {
                "schema_version": DRIVER_SCHEMA_VERSION,
                "reequation_source": dict(plan.source_record),
            })
    for scenario_id, corpus in plan.corpora.items():
        materialize(scenarios=scenario_id, generated=corpus, **shared)


def _all_configs_exist(plan: Plan) -> bool:
    return all(cell.config_path(plan.bench, variant).is_file() for cell, variant in plan.runs)


def run_configs(plan: Plan, *, base: Path, confirmatory: bool, force: bool, seeds: str | None,
                sizes: str | None, registry_path: Path | None, seeds_path: Path | None,
                ledger_path: Path | None) -> None:
    if confirmatory and _all_configs_exist(plan) and not force:
        # The confirmatory configurations were written, and logged in the
        # ledger, by an earlier invocation; this one resumes their runs.
        print("Confirmatory configurations exist; resuming their runs")
        return
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = write_configs(
            base=base, out=plan.bench, scenarios=",".join(s.id for s in plan.scenarios),
            sizes=sizes, seeds=seeds, confirmatory=confirmatory, force=force,
            registry_path=registry_path, seeds_path=seeds_path, ledger_path=ledger_path,
        )
    log = plan.bench / "driver" / "logs" / "configs.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n# {_now()}\n{buffer.getvalue()}")
    print(f"Wrote {len(result['runs'])} run configuration(s) from {base} (commands in {log})")


def run_cells(
    plan: Plan,
    environ: Mapping[str, str],
    *,
    runner: Runner = run_logged,
    stop_on_failure: bool = False,
) -> list[dict[str, Any]]:
    runs = plan.runs
    results: list[dict[str, Any]] = []
    for index, (cell, variant) in enumerate(runs, start=1):
        run_dir = cell.run_dir(plan.bench, variant)
        status, detail = run_status(run_dir)
        if status == "complete":
            results.append({"cell": cell.id, "variant": variant, "result": "complete"})
            continue
        config = cell.config_path(plan.bench, variant)
        log = plan.bench / "driver" / "logs" / f"run_{cell.id.replace('/', '_')}_{variant}.log"
        print(f"\n== run {index}/{len(runs)} {cell.id} {variant} ({detail})\n"
              f"   output: {run_dir}\n   log: {log}")
        env = {"PRESET": plan.machine, **environ, "OCI_PYTHON": sys.executable}
        code = runner(["bash", str(LAUNCHER), str(config)], log, env)
        status, detail = run_status(run_dir)
        result = "complete" if status == "complete" else "failed"
        results.append({"cell": cell.id, "variant": variant, "result": result,
                        "exit_code": code, "detail": detail})
        print(f"== {cell.id} {variant}: {result} ({detail}, exit code {code})")
        if result == "failed" and stop_on_failure:
            break
    return results


# ---------------------------------------------------------------------------
# command line


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_benchmark.sh",
        description=(
            "Run or resume the benchmark end to end on one machine: LLM corpora where the "
            "selected cells need them, cohorts, run configurations, one workflow run per "
            "cell (one at a time, smallest cells first), and the aggregate report. Rerun "
            "the same command to resume."
        ),
    )
    parser.add_argument("machine", help=f"hardware preset in scripts/presets/ ({', '.join(machines())})")
    parser.add_argument("out", type=Path, help="benchmark directory BENCH")
    parser.add_argument("--scenarios", help="scenario ids or families, comma-separated (default: all)")
    parser.add_argument("--sizes", help="comma-separated sizes (default: each scenario's sizes)")
    parser.add_argument("--seeds", help="comma-separated seeds, or 'development' / 'confirmatory' "
                        "(default: the development seeds)")
    parser.add_argument("--confirmatory", action="store_true",
                        help="use confirmatory seeds; their configurations are logged in the ledger once")
    parser.add_argument("--force", action="store_true",
                        help="write confirmatory configurations although the ledger has this commit")
    parser.add_argument("--source", type=Path,
                        help="re-equation source for S0-S4 (default: the bundled five-confounder cohort)")
    parser.add_argument("--source-size", type=int,
                        help="expand the source to this many patients even if the selected cells "
                        "need fewer, so later selections can use the same source")
    parser.add_argument("--llm-url",
                        help="comma-separated URLs of running vLLM (OpenAI-compatible) servers that "
                        "serve one model; they replace the preset's managed servers for both Stage 2 "
                        "roles and the generator, and the preset keeps only its hardware settings")
    parser.add_argument("--llm-concurrency", type=int, default=32,
                        help="requests at a time per --llm-url server, shared by interpretation "
                        "(a quarter on the first server) and extraction (default: 32)")
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE,
                        help="base workflow configuration (default: benchmarks/base_config.json)")
    parser.add_argument("--through", choices=STEPS, default="aggregate",
                        help="stop after this step (default: aggregate)")
    parser.add_argument("--plan", action="store_true",
                        help="print the plan and the status of each step, then stop")
    parser.add_argument("--stop-on-failure", action="store_true",
                        help="stop at the first failed run instead of continuing with the next")
    parser.add_argument("--no-ceiling", action="store_true", help="skip the oracle-covariate ceiling")
    parser.add_argument("--ceiling-workers", type=int, default=1,
                        help="CPU processes for the ceiling refits (default: 1)")
    parser.add_argument("--registry", type=Path, help="scenario registry (default: benchmarks/scenarios.json)")
    parser.add_argument("--seeds-file", type=Path, help="seed file (default: benchmarks/seeds.json)")
    parser.add_argument("--ledger", type=Path,
                        help="confirmatory ledger (default: benchmarks/confirmatory_ledger.jsonl)")
    return parser


def run(args: argparse.Namespace, *, environ: Mapping[str, str] | None = None,
        runner: Runner = run_logged, endpoints_context: Callable[..., Any] = generation_endpoints,
        fetch: Callable[[str], Any] = _fetch_models) -> int:
    environ = dict(os.environ if environ is None else environ)
    env = preset_environment(args.machine, environ)
    servers = (probe_llm_servers(args.llm_url, concurrency=args.llm_concurrency, fetch=fetch)
               if args.llm_url else None)
    if servers is not None:
        run_env = external_run_environment(environ, env, servers)
        env = generator_environment(env, servers)
    else:
        run_env = {**environ, "PRESET": args.machine}
    plan = build_plan(
        machine=args.machine, bench=args.out, scenarios=args.scenarios, sizes=args.sizes,
        seeds=args.seeds, confirmatory=args.confirmatory, source=args.source,
        source_size=args.source_size, registry_path=args.registry, seeds_path=args.seeds_file,
    )
    print(describe(plan, env, servers))
    if args.plan:
        return 0
    base = Path(args.base).expanduser().resolve()
    if not base.is_file():
        raise DriverError(f"base configuration {base} does not exist")
    log = plan.bench / "driver" / "invocations.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "schema_version": DRIVER_SCHEMA_VERSION, "time": _now(), "machine": args.machine,
            "argv": [str(part) for part in sys.argv[1:]], "git": git_state(),
            "stage2_model": run_env.get("STAGE2_MODEL") or env.get("STAGE2_MODEL"),
            "stage2_extraction_model": (run_env.get("STAGE2_EXTRACTION_MODEL")
                                        or env.get("STAGE2_EXTRACTION_MODEL")),
            "generator_model": _generator_model(env) if plan.jobs else None,
            "llm_servers": None if servers is None else {
                "urls": list(servers.urls), "model": servers.model,
                "context_window": servers.context_window, "concurrency": servers.concurrency,
            },
        }, sort_keys=True) + "\n")
    selection = {"sizes": args.sizes, "seeds": args.seeds, "registry_path": args.registry,
                 "seeds_path": args.seeds_file}
    last = STEPS.index(args.through)
    results: list[dict[str, Any]] = []
    run_generation(plan, env, runner=runner, endpoints_context=endpoints_context,
                   max_tokens=generator_max_tokens(servers, env))
    if last >= STEPS.index("materialize"):
        print("\n== materialize")
        run_materialize(plan, confirmatory=args.confirmatory, **selection)
    if last >= STEPS.index("configs"):
        print("\n== configs")
        run_configs(plan, base=base, confirmatory=args.confirmatory, force=args.force,
                    ledger_path=args.ledger, **selection)
    if last >= STEPS.index("runs"):
        results = run_cells(plan, run_env, runner=runner, stop_on_failure=args.stop_on_failure)
    if last >= STEPS.index("aggregate"):
        print("\n== aggregate")
        aggregate(out=plan.bench, scenarios=",".join(s.id for s in plan.scenarios),
                  sizes=args.sizes, seeds=args.seeds, confirmatory=args.confirmatory,
                  ceiling=not args.no_ceiling, ceiling_workers=args.ceiling_workers,
                  registry_path=args.registry, seeds_path=args.seeds_file)
        print(f"Report: {plan.bench / 'aggregate' / 'summary.md'}")
    failed = [item for item in results if item["result"] == "failed"]
    if results:
        print(f"\nRuns: {sum(item['result'] == 'complete' for item in results)} complete, "
              f"{len(failed)} failed")
        for item in failed:
            print(f"  failed: {item['cell']} {item['variant']} ({item['detail']})")
    return 1 if failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except SeedGuardError as error:
        print(f"refused: {error}")
        return 2
    except (DriverError, FileExistsError, FileNotFoundError, ValueError) as error:
        print(f"error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
