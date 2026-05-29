#!/usr/bin/env python
"""Expanded oracle runner for agentic explicit-feature forests only.

This is a constrained version of run_oracle_experiments.py. It always runs
only the agentic_explicit_feature_forest method, but uses a broader default
grid and more repeats:

    base conditions per dataset:
      agentic_iterations: 1, 2, 3
      initial_feature_counts: 0, 2, 5
      initial_feature_strategies: true_first, modifiers_first, mixed, distractors
      stop_after_rejected_iteration: true, false

The count=0 condition uses the "none" strategy internally, so this produces
54 base conditions per dataset. With the default 30 repeats, that is 1620
experiments per dataset before --max-experiments or --resume filtering.

The default LLM budgets are intentionally large:
    agent proposal context chars: 200000
    agent proposal generation tokens: 50000
    extraction text chars: 200000
    extraction generation tokens: 50000

By default this wrapper starts one local vLLM OpenAI-compatible server per
requested CUDA device, using sequential ports starting at 8000. Wrapper-only
vLLM server options include --download-dir, --gpu-memory-utilization,
--max-num-seqs, --max-num-batched-tokens, --dtype, --kv-cache-dtype,
--vllm-port-base, --vllm-extra-arg, --vllm-extra-args, and --no-start-vllm.
"""

import sys
import atexit
import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_oracle_experiments as oracle_runner


DEFAULT_ARGS: Dict[str, List[str]] = {
    "--output-dir": ["../pcori_experiments/oracle_agentic_explicit_forest_expanded"],
    "--n-repeats": ["30"],
    "--agentic-iterations": ["1", "2", "3"],
    "--agentic-initial-feature-counts": ["0", "2", "5"],
    "--agentic-initial-feature-strategies": [
        "true_first",
        "modifiers_first",
        "mixed",
        "distractors",
    ],
    "--agentic-inner-folds": ["3"],
    "--agentic-max-additions-per-iter": ["6"],
    "--agentic-max-removals-per-iter": ["2"],
    "--agentic-stop-after-rejected-iteration": ["true", "false"],
    "--agentic-agent-model-name": ["nvidia/Gemma-4-31B-IT-NVFP4"],
    "--agentic-agent-server-url": ["http://localhost:8000/v1"],
    "--agentic-vllm-model-name": ["nvidia/Gemma-4-31B-IT-NVFP4"],
    "--agentic-vllm-server-url": ["http://localhost:8000/v1"],
    "--agentic-agent-max-tokens": ["50000"],
    "--agentic-agent-context-chars": ["200000"],
    "--agentic-agent-context-examples": ["20"],
    "--agentic-vllm-max-model-len": ["200000"],
    "--agentic-extraction-max-tokens": ["50000"],
    "--agentic-extraction-max-text-length": ["200000"],
    "--agentic-extraction-max-retries": ["5"],
    "--agentic-extraction-batch-size": ["16"],
}

WRAPPER_VLLM_VALUE_OPTIONS = {
    "--download-dir": "download_dir",
    "--vllm-download-dir": "download_dir",
    "--gpu-memory-utilization": "gpu_memory_utilization",
    "--vllm-gpu-memory-utilization": "gpu_memory_utilization",
    "--max-num-seqs": "max_num_seqs",
    "--vllm-max-num-seqs": "max_num_seqs",
    "--max-num-batched-tokens": "max_num_batched_tokens",
    "--vllm-max-num-batched-tokens": "max_num_batched_tokens",
    "--dtype": "dtype",
    "--vllm-dtype": "dtype",
    "--kv-cache-dtype": "kv_cache_dtype",
    "--vllm-kv-cache-dtype": "kv_cache_dtype",
    "--port-base": "port_base",
    "--vllm-port-base": "port_base",
    "--startup-timeout": "startup_timeout",
    "--vllm-startup-timeout": "startup_timeout",
}

WRAPPER_VLLM_FLAG_OPTIONS = {
    "--enforce-eager",
    "--disable-custom-all-reduce",
}

FORCED_AGENTIC_ONLY_ARGS = [
    "--model-types",
    "agentic_explicit_feature_forest",
    "--filter-extractor-types",
    "agentic_explicit_features",
]


def _option_present(args: List[str], option: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in args)


def _option_value(args: List[str], option: str, default: Optional[str] = None) -> Optional[str]:
    for idx, arg in enumerate(args):
        if arg.startswith(f"{option}="):
            return arg.split("=", 1)[1]
        if arg == option and idx + 1 < len(args):
            return args[idx + 1]
    return default


def _option_values(args: List[str], option: str, default: Optional[List[str]] = None) -> List[str]:
    for idx, arg in enumerate(args):
        if arg.startswith(f"{option}="):
            return [arg.split("=", 1)[1]]
        if arg == option:
            values = []
            for value in args[idx + 1:]:
                if value.startswith("--"):
                    break
                values.append(value)
            return values
    return list(default or [])


def _extract_wrapper_vllm_args(argv: List[str]) -> tuple[List[str], Dict[str, Any]]:
    """Remove wrapper-only vLLM server args before oracle argparse sees argv."""
    cleaned = [argv[0]]
    settings: Dict[str, Any] = {
        "start_vllm": True,
        "download_dir": None,
        "gpu_memory_utilization": "0.95",
        "max_num_seqs": "4",
        "max_num_batched_tokens": None,
        "dtype": None,
        "kv_cache_dtype": None,
        "port_base": None,
        "startup_timeout": 1200,
        "extra_args": [],
    }

    idx = 1
    while idx < len(argv):
        arg = argv[idx]
        name, has_inline_value, inline_value = arg.partition("=")

        if name in WRAPPER_VLLM_VALUE_OPTIONS:
            key = WRAPPER_VLLM_VALUE_OPTIONS[name]
            if has_inline_value:
                value = inline_value
            else:
                idx += 1
                if idx >= len(argv):
                    raise ValueError(f"{name} requires a value")
                value = argv[idx]
            settings[key] = value
        elif name in WRAPPER_VLLM_FLAG_OPTIONS and not has_inline_value:
            settings["extra_args"].append(name)
        elif name == "--vllm-extra-arg":
            if has_inline_value:
                settings["extra_args"].append(inline_value)
            else:
                idx += 1
                if idx >= len(argv):
                    raise ValueError("--vllm-extra-arg requires a value")
                settings["extra_args"].append(argv[idx])
        elif name == "--vllm-extra-args":
            if has_inline_value:
                value = inline_value
            else:
                idx += 1
                if idx >= len(argv):
                    raise ValueError("--vllm-extra-args requires a value")
                value = argv[idx]
            settings["extra_args"].extend(shlex.split(value))
        elif name == "--no-start-vllm" and not has_inline_value:
            settings["start_vllm"] = False
        else:
            cleaned.append(arg)
        idx += 1

    if (
        settings["download_dir"]
        and not _option_present(cleaned[1:], "--agentic-vllm-download-dir")
    ):
        cleaned.extend(["--agentic-vllm-download-dir", str(settings["download_dir"])])

    return cleaned, settings


def _with_expanded_agentic_defaults(argv: List[str]) -> List[str]:
    args = list(argv[1:])
    expanded_args = list(args)

    for option, values in DEFAULT_ARGS.items():
        if not _option_present(args, option):
            expanded_args.extend([option, *values])

    # Keep this last so this runner cannot accidentally launch the full oracle
    # model grid if a caller passes --model-types or --filter-extractor-types.
    expanded_args.extend(FORCED_AGENTIC_ONLY_ARGS)
    return [argv[0], *expanded_args]


def _health_url(server_url: str) -> str:
    parsed = urlparse(server_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return parsed._replace(path=f"{path}/health", params="", query="", fragment="").geturl()


def _server_is_reachable(server_url: str) -> bool:
    try:
        with urlopen(_health_url(server_url), timeout=2) as response:
            return response.status == 200
    except URLError:
        return False
    except TimeoutError:
        return False


def _is_local_server(server_url: str) -> bool:
    host = urlparse(server_url).hostname
    return host in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


def _cuda_device_id(device: str) -> Optional[str]:
    if device.startswith("cuda:"):
        return device.split(":", 1)[1]
    return None


def _server_url_for_port(server_url: str, port: int) -> str:
    parsed = urlparse(server_url)
    host = parsed.hostname or "localhost"
    if host == "0.0.0.0":
        host = "localhost"
    return parsed._replace(netloc=f"{host}:{port}", path="/v1", params="", query="", fragment="").geturl()


def _vllm_cmd(
    *,
    server_url: str,
    model_name: str,
    max_model_len: str,
    settings: Dict[str, Any],
) -> List[str]:
    parsed = urlparse(server_url)
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(model_name),
        "--served-model-name",
        str(model_name),
        "--host",
        parsed.hostname or "localhost",
        "--port",
        str(parsed.port or 8000),
        "--tensor-parallel-size",
        "1",
        "--gpu-memory-utilization",
        str(settings["gpu_memory_utilization"]),
        "--max-model-len",
        str(max_model_len),
        "--max-num-seqs",
        str(settings["max_num_seqs"]),
        "--trust-remote-code",
    ]
    if settings["download_dir"]:
        cmd.extend(["--download-dir", str(settings["download_dir"])])
    if settings["max_num_batched_tokens"]:
        cmd.extend(["--max-num-batched-tokens", str(settings["max_num_batched_tokens"])])
    if settings["dtype"]:
        cmd.extend(["--dtype", str(settings["dtype"])])
    if settings["kv_cache_dtype"]:
        cmd.extend(["--kv-cache-dtype", str(settings["kv_cache_dtype"])])
    cmd.extend(settings["extra_args"])
    return cmd


def _start_local_vllm_servers(argv: List[str], settings: Dict[str, Any]) -> None:
    args = argv[1:]
    server_url = _option_value(args, "--agentic-agent-server-url", "http://localhost:8000/v1")
    if not server_url or not _is_local_server(server_url):
        return

    parsed = urlparse(server_url)
    model_name = _option_value(args, "--agentic-agent-model-name", "nvidia/Gemma-4-31B-IT-NVFP4")
    max_model_len = _option_value(args, "--agentic-vllm-max-model-len", "200000")
    base_port = int(settings["port_base"] or parsed.port or 8000)
    output_dir = Path(_option_value(
        args,
        "--output-dir",
        "../pcori_experiments/oracle_agentic_explicit_forest_expanded",
    ))
    output_dir.mkdir(parents=True, exist_ok=True)

    devices = _option_values(args, "--devices", ["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
    if not devices:
        devices = ["cuda:0"]

    url_by_device: Dict[str, str] = {}
    processes = []
    log_files = []

    for idx, device in enumerate(devices):
        device_id = _cuda_device_id(device)
        port = base_port + idx
        device_server_url = _server_url_for_port(server_url, port)
        url_by_device[device] = device_server_url

        if _server_is_reachable(device_server_url):
            print(f"Using existing vLLM server for {device}: {device_server_url}")
            continue

        log_suffix = f"cuda_{device_id}" if device_id is not None else f"device_{idx}"
        log_path = output_dir / f"vllm_server_{log_suffix}.log"
        cmd = _vllm_cmd(
            server_url=device_server_url,
            model_name=str(model_name),
            max_model_len=str(max_model_len),
            settings=settings,
        )
        env = os.environ.copy()
        if device_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = device_id

        print(f"Starting local vLLM server for {device}: {device_server_url}")
        print(f"vLLM log: {log_path}")
        log_file = open(log_path, "a")
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
        )
        processes.append((process, log_path))
        log_files.append(log_file)

        deadline = time.time() + int(settings["startup_timeout"])
        while time.time() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"vLLM server for {device} exited before becoming ready. See {log_path}"
                )
            if _server_is_reachable(device_server_url):
                print(f"Local vLLM server for {device} is ready.")
                break
            time.sleep(5)
        else:
            raise TimeoutError(
                f"Timed out waiting for vLLM server for {device}. See {log_path}"
            )

    os.environ["OCI_AGENTIC_AGENT_SERVER_URLS_BY_DEVICE"] = json.dumps(url_by_device)
    os.environ["OCI_AGENTIC_VLLM_SERVER_URLS_BY_DEVICE"] = json.dumps(url_by_device)

    def _cleanup() -> None:
        for process, _log_path in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
        for log_file in log_files:
            log_file.close()

    atexit.register(_cleanup)


def main() -> None:
    wants_help = "-h" in sys.argv[1:] or "--help" in sys.argv[1:]
    cleaned_argv, vllm_settings = _extract_wrapper_vllm_args(sys.argv)
    sys.argv = _with_expanded_agentic_defaults(cleaned_argv)
    if not wants_help and vllm_settings["start_vllm"]:
        _start_local_vllm_servers(sys.argv, vllm_settings)
    oracle_runner.main()


if __name__ == "__main__":
    main()
