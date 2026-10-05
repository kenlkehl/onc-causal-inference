"""Lifecycle management for a local pool of OpenAI-compatible vLLM servers."""

from __future__ import annotations

import json
import logging
import math
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

LOGGER = logging.getLogger(__name__)

_RESERVED_EXTRA_ARGUMENTS = {
    "--api-key",
    "--default-chat-template-kwargs",
    "--download-dir",
    "--host",
    "--language-model-only",
    "--model",
    "--no-language-model-only",
    "--port",
    "--reasoning-parser",
    "--served-model-name",
    "--tensor-parallel-size",
    "-tp",
}

# vLLM briefly probes and releases internal rendezvous ports before its engine
# subprocess binds them.  Replicas started together can therefore select the
# same ephemeral port.  Give every managed replica a disjoint starting range;
# vLLM will scan upward within that range when a port is already occupied.
_VLLM_INTERNAL_PORT_BASE = 20_000
_VLLM_INTERNAL_PORT_STRIDE = 128


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _normalized_gpu(value: Any) -> str:
    text = str(value).strip().lower()
    if text.startswith("cuda:"):
        text = text.split(":", 1)[1]
    if not text.isdigit():
        raise ValueError(
            "stage2.vllm.gpus entries must be logical CUDA indices such as 0 or cuda:0"
        )
    return f"cuda:{int(text)}"


def _model_family_defaults(model: str) -> tuple[str, bool | None, Mapping[str, Any] | None]:
    """Return requested vLLM defaults for recognized reasoning-model families."""

    normalized = str(model).strip().lower()
    if "gemma" in normalized:
        # Thinking is selected per request through reasoning_effort. A server
        # default would force extraction and interpretation into one mode.
        return "gemma4", True, None
    if "qwen" in normalized:
        return "qwen3", True, None
    return "", None, None


def _vllm_internal_port_bases(
    server_count: int,
    *,
    internal_port_base: int = _VLLM_INTERNAL_PORT_BASE,
) -> tuple[int, ...]:
    bases = tuple(
        int(internal_port_base) + index * _VLLM_INTERNAL_PORT_STRIDE
        for index in range(int(server_count))
    )
    if bases and bases[0] < 1:
        raise ValueError("stage2.vllm.internal_port_base must be between 1 and 65535")
    if bases and bases[-1] + _VLLM_INTERNAL_PORT_STRIDE - 1 > 65_535:
        raise ValueError(
            "stage2.vllm.server_count is too large to allocate disjoint internal "
            "vLLM port ranges"
        )
    return bases


def validate_managed_vllm_pool_isolation(
    primary: ManagedVLLMConfig,
    extraction: ManagedVLLMConfig,
) -> None:
    """Reject internal rendezvous ranges shared by concurrently resident pools."""

    primary.validate()
    extraction.validate()
    primary_bases = _vllm_internal_port_bases(
        primary.server_count,
        internal_port_base=primary.internal_port_base,
    )
    extraction_bases = _vllm_internal_port_bases(
        extraction.server_count,
        internal_port_base=extraction.internal_port_base,
    )
    overlaps = [
        (primary_base, extraction_base)
        for primary_base in primary_bases
        for extraction_base in extraction_bases
        if max(primary_base, extraction_base)
        <= min(
            primary_base + _VLLM_INTERNAL_PORT_STRIDE - 1,
            extraction_base + _VLLM_INTERNAL_PORT_STRIDE - 1,
        )
    ]
    if overlaps:
        raise ValueError(
            "primary and extraction managed vLLM pools have overlapping internal "
            "rendezvous ranges; configure distinct internal_port_base values "
            f"(overlapping replica bases={overlaps})"
        )


def managed_argument(config, flag, default=None):
    """Read the effective last occurrence of a vLLM extra argument."""
    value = default
    for index, token in enumerate(config.extra_args):
        if token.startswith(flag + "="):
            value = token.split("=", 1)[1]
        elif token == flag:
            if index + 1 >= len(config.extra_args):
                raise ValueError(f"{flag} requires a value")
            value = config.extra_args[index + 1]
    return value


def co_resident_vllm_configs(primary, extraction):
    """Replicate both roles on their configured GPU union, preserving TP widths.

    Explicit memory fractions reserve room for the other role and retrieval.
    The GPU union respects CUDA visibility and the run's selected devices; no
    unrelated host GPU is claimed. Port expansion keeps both pools isolated.
    """
    primary.validate()
    extraction.validate()
    fractions = []
    for config in (primary, extraction):
        raw = managed_argument(config, "--gpu-memory-utilization")
        if raw is None:
            raise ValueError("co-resident all-GPU serving requires explicit --gpu-memory-utilization for both models")
        fraction = float(raw)
        if not math.isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError("co-resident GPU memory fractions must be finite and positive")
        fractions.append(fraction)
    if sum(fractions) > 0.90 + 1e-12:
        raise ValueError("co-resident GPU memory fractions must total at most 0.90, leaving room for retrieval and CUDA contexts")
    gpus = tuple(dict.fromkeys((*primary.gpus, *extraction.gpus)))
    ports = set(primary.effective_ports())
    other_ports = set(extraction.effective_ports())
    if ports & other_ports:
        raise ValueError("co-resident model pools require distinct public HTTP ports")
    used = ports | other_ports

    def expand(config):
        width = config.effective_gpus_per_server()
        if len(gpus) % width:
            raise ValueError("co-resident GPU union must divide evenly by each model's tensor-parallel width")
        count = len(gpus) // width
        expanded_ports = list(config.effective_ports())
        candidate = max(expanded_ports) + 1
        while len(expanded_ports) < count:
            if candidate > 65535:
                raise ValueError("cannot allocate co-resident model ports below 65536")
            if candidate not in used:
                expanded_ports.append(candidate)
                used.add(candidate)
            candidate += 1
        result = replace(config, gpus=gpus, server_count=count,
                         gpus_per_server=width, ports=tuple(expanded_ports))
        result.validate()
        return result

    primary, extraction = expand(primary), expand(extraction)
    # Expanding a pool may overtake the other pool's original rendezvous range.
    # Preserve its base when possible, otherwise allocate after the primary.
    primary_end = primary.internal_port_base + primary.server_count * _VLLM_INTERNAL_PORT_STRIDE
    extraction_end = extraction.internal_port_base + extraction.server_count * _VLLM_INTERNAL_PORT_STRIDE
    if max(primary.internal_port_base, extraction.internal_port_base) < min(primary_end, extraction_end):
        extraction = replace(extraction, internal_port_base=primary_end)
    validate_managed_vllm_pool_isolation(primary, extraction)
    return primary, extraction


@dataclass(frozen=True)
class ManagedVLLMConfig:
    """Configuration for pipeline-owned vLLM replica processes."""

    server_count: int
    gpus: tuple[str, ...]
    # When supplied, this is an explicit tensor-parallel width and must agree
    # with server_count. The mapping parser can derive server_count from it.
    gpus_per_server: int | None = None
    host: str = "127.0.0.1"
    base_port: int = 8010
    ports: tuple[int, ...] = ()
    # Distinct pools must use distinct ranges because vLLM engine subprocesses
    # bind rendezvous ports independently of their public HTTP ports.
    internal_port_base: int = _VLLM_INTERNAL_PORT_BASE
    startup_timeout: float = 7_200.0
    startup_poll_interval: float = 2.0
    shutdown_timeout: float = 30.0
    download_dir: str = ""
    reasoning_parser: str = ""
    language_model_only: bool | None = None
    default_chat_template_kwargs: Mapping[str, Any] | None = None
    extra_args: tuple[str, ...] = ()

    def validate(self) -> None:
        if self.server_count < 1:
            raise ValueError("stage2.vllm.server_count must be positive")
        if not self.gpus:
            raise ValueError("stage2.vllm.gpus must contain at least one logical CUDA device")
        if len(self.gpus) != len(set(self.gpus)):
            raise ValueError("stage2.vllm.gpus must not contain duplicate devices")
        if self.server_count > len(self.gpus):
            raise ValueError(
                "stage2.vllm.server_count cannot exceed the number of supplied GPUs"
            )
        if len(self.gpus) % self.server_count:
            raise ValueError(
                "stage2.vllm.gpus must divide evenly across stage2.vllm.server_count"
            )
        effective_gpus_per_server = len(self.gpus) // self.server_count
        if self.gpus_per_server is not None:
            if (
                isinstance(self.gpus_per_server, bool)
                or not isinstance(self.gpus_per_server, int)
                or self.gpus_per_server < 1
            ):
                raise ValueError("stage2.vllm.gpus_per_server must be a positive integer")
            if self.gpus_per_server != effective_gpus_per_server:
                raise ValueError(
                    "stage2.vllm.gpus_per_server must agree with the supplied GPU list "
                    "and stage2.vllm.server_count"
                )
        if not str(self.host).strip():
            raise ValueError("stage2.vllm.host must be nonempty")
        effective_ports = self.effective_ports()
        if len(effective_ports) != len(set(effective_ports)):
            raise ValueError("stage2.vllm ports must be unique")
        if any(port < 1 or port > 65_535 for port in effective_ports):
            raise ValueError("stage2.vllm ports must be between 1 and 65535")
        _vllm_internal_port_bases(
            self.server_count,
            internal_port_base=self.internal_port_base,
        )
        if self.startup_timeout <= 0:
            raise ValueError("stage2.vllm.startup_timeout must be positive")
        if self.startup_poll_interval <= 0:
            raise ValueError("stage2.vllm.startup_poll_interval must be positive")
        if self.shutdown_timeout <= 0:
            raise ValueError("stage2.vllm.shutdown_timeout must be positive")
        if self.language_model_only is not None and not isinstance(
            self.language_model_only, bool
        ):
            raise ValueError("stage2.vllm.language_model_only must be true, false, or null")
        if self.default_chat_template_kwargs is not None and not isinstance(
            self.default_chat_template_kwargs, Mapping
        ):
            raise ValueError("stage2.vllm.default_chat_template_kwargs must be an object or null")
        for argument in self.extra_args:
            token = str(argument)
            flag = token.split("=", 1)[0]
            if flag in _RESERVED_EXTRA_ARGUMENTS:
                raise ValueError(
                    f"stage2.vllm.extra_args cannot override managed argument {flag}; "
                    "use its named stage2.vllm setting instead"
                )

    def effective_ports(self) -> tuple[int, ...]:
        if self.ports:
            if len(self.ports) != self.server_count:
                raise ValueError(
                    "stage2.vllm.ports must contain exactly stage2.vllm.server_count ports"
                )
            return tuple(int(port) for port in self.ports)
        return tuple(int(self.base_port) + index for index in range(self.server_count))

    def gpu_groups(self) -> tuple[tuple[str, ...], ...]:
        self.validate()
        per_server = self.effective_gpus_per_server()
        return tuple(
            tuple(self.gpus[start : start + per_server])
            for start in range(0, len(self.gpus), per_server)
        )

    def effective_gpus_per_server(self) -> int:
        return len(self.gpus) // self.server_count

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


def managed_vllm_config_from_mapping(
    raw: Any,
    *,
    model: str,
    default_base_port: int = 8010,
    default_internal_port_base: int = _VLLM_INTERNAL_PORT_BASE,
) -> ManagedVLLMConfig | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("stage2.vllm must be an object")
    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("stage2.vllm.enabled must be true or false")
    if not enabled:
        return None

    raw_gpus = raw.get("gpus")
    if isinstance(raw_gpus, str):
        raw_gpus = [part.strip() for part in raw_gpus.split(",") if part.strip()]
    if not isinstance(raw_gpus, Sequence) or isinstance(raw_gpus, (bytes, bytearray)):
        raise ValueError("stage2.vllm.gpus must be a list or comma-separated string")
    gpus = tuple(_normalized_gpu(value) for value in raw_gpus)

    raw_server_count = raw.get("server_count", raw.get("servers"))
    raw_gpus_per_server = raw.get("gpus_per_server")
    gpus_per_server = (
        None if raw_gpus_per_server is None else int(raw_gpus_per_server)
    )
    if gpus_per_server is not None:
        if gpus_per_server < 1:
            raise ValueError("stage2.vllm.gpus_per_server must be a positive integer")
        if len(gpus) % gpus_per_server:
            raise ValueError(
                "stage2.vllm.gpus must divide evenly across "
                "stage2.vllm.gpus_per_server"
            )
        derived_server_count = len(gpus) // gpus_per_server
        if raw_server_count is not None and int(raw_server_count) != derived_server_count:
            raise ValueError(
                "stage2.vllm.server_count and stage2.vllm.gpus_per_server disagree "
                "for the supplied GPU list"
            )
        server_count = derived_server_count
    else:
        server_count = len(gpus) if raw_server_count is None else int(raw_server_count)

    raw_ports = raw.get("ports", ())
    if isinstance(raw_ports, str):
        raw_ports = [part.strip() for part in raw_ports.split(",") if part.strip()]
    if not isinstance(raw_ports, Sequence) or isinstance(raw_ports, (bytes, bytearray)):
        raise ValueError("stage2.vllm.ports must be a list or comma-separated string")

    raw_extra_args = raw.get("extra_args", ())
    if isinstance(raw_extra_args, str):
        raise ValueError(
            "stage2.vllm.extra_args must be a list of individual command-line tokens"
        )
    if not isinstance(raw_extra_args, Sequence) or isinstance(
        raw_extra_args, (bytes, bytearray)
    ):
        raise ValueError("stage2.vllm.extra_args must be a list")

    default_parser, default_language_only, default_chat_kwargs = _model_family_defaults(model)
    reasoning_parser = (
        default_parser
        if "reasoning_parser" not in raw
        else str(raw.get("reasoning_parser") or "").strip()
    )
    language_model_only = (
        default_language_only
        if "language_model_only" not in raw
        else raw.get("language_model_only")
    )
    default_chat_template_kwargs = (
        default_chat_kwargs
        if "default_chat_template_kwargs" not in raw
        else raw.get("default_chat_template_kwargs")
    )

    config = ManagedVLLMConfig(
        server_count=server_count,
        gpus=gpus,
        gpus_per_server=gpus_per_server,
        host=str(raw.get("host", "127.0.0.1")).strip(),
        base_port=int(raw.get("base_port", default_base_port)),
        ports=tuple(int(port) for port in raw_ports),
        internal_port_base=int(
            raw.get("internal_port_base", default_internal_port_base)
        ),
        startup_timeout=float(raw.get("startup_timeout", 7_200.0)),
        startup_poll_interval=float(raw.get("startup_poll_interval", 2.0)),
        shutdown_timeout=float(raw.get("shutdown_timeout", 30.0)),
        download_dir=str(raw.get("download_dir") or "").strip(),
        reasoning_parser=reasoning_parser,
        language_model_only=language_model_only,
        default_chat_template_kwargs=default_chat_template_kwargs,
        extra_args=tuple(str(argument) for argument in raw_extra_args),
    )
    config.validate()
    return config


def _visible_gpu_tokens(
    group: Sequence[str],
    *,
    parent_cuda_visible_devices: str | None,
) -> tuple[str, ...]:
    logical_indices = [int(str(device).split(":", 1)[1]) for device in group]
    if parent_cuda_visible_devices is None:
        return tuple(str(index) for index in logical_indices)
    parent_tokens = [
        token.strip() for token in parent_cuda_visible_devices.split(",") if token.strip()
    ]
    if not parent_tokens or parent_tokens == ["-1"]:
        raise ValueError(
            "stage2.vllm cannot launch GPU servers because CUDA_VISIBLE_DEVICES hides all GPUs"
        )
    missing = [index for index in logical_indices if index >= len(parent_tokens)]
    if missing:
        raise ValueError(
            "stage2.vllm.gpus contains logical indices outside CUDA_VISIBLE_DEVICES: "
            f"{missing}; visible logical range is 0..{len(parent_tokens) - 1}"
        )
    return tuple(parent_tokens[index] for index in logical_indices)


def _client_host(host: str) -> str:
    normalized = str(host).strip()
    if normalized in {"0.0.0.0", "::"}:
        return "127.0.0.1" if normalized == "0.0.0.0" else "::1"
    return normalized


def _endpoint(host: str, port: int) -> str:
    client_host = _client_host(host)
    rendered_host = f"[{client_host}]" if ":" in client_host else client_host
    return f"http://{rendered_host}:{int(port)}/v1"


def _build_vllm_command(
    config: ManagedVLLMConfig,
    *,
    model: str,
    api_key: str,
    port: int,
    tensor_parallel_size: int,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        str(model),
        "--host",
        config.host,
        "--port",
        str(int(port)),
        "--tensor-parallel-size",
        str(int(tensor_parallel_size)),
    ]
    if config.download_dir:
        command.extend(["--download-dir", config.download_dir])
    if config.reasoning_parser:
        command.extend(["--reasoning-parser", config.reasoning_parser])
    if config.language_model_only is True:
        command.append("--language-model-only")
    elif config.language_model_only is False:
        command.append("--no-language-model-only")
    if config.default_chat_template_kwargs is not None:
        command.extend(
            [
                "--default-chat-template-kwargs",
                json.dumps(
                    dict(config.default_chat_template_kwargs),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ]
        )
    if api_key and api_key != "EMPTY":
        command.extend(["--api-key", api_key])
    command.extend(config.extra_args)
    return command


def _redacted_command(command: Sequence[str]) -> list[str]:
    redacted = list(command)
    for index, token in enumerate(redacted):
        if token == "--api-key" and index + 1 < len(redacted):
            redacted[index + 1] = "<redacted>"
        elif token.startswith("--api-key="):
            redacted[index] = "--api-key=<redacted>"
    return redacted


def _assert_port_available(host: str, port: int) -> None:
    bind_host = str(host).strip()
    try:
        addresses = socket.getaddrinfo(
            bind_host,
            int(port),
            type=socket.SOCK_STREAM,
            flags=socket.AI_PASSIVE,
        )
    except socket.gaierror as exc:
        raise ValueError(f"stage2.vllm.host could not be resolved: {host!r}") from exc
    last_error: OSError | None = None
    for family, socktype, proto, _canonical_name, sockaddr in addresses:
        probe = socket.socket(family, socktype, proto)
        try:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(sockaddr)
            return
        except OSError as exc:
            last_error = exc
        finally:
            probe.close()
    raise RuntimeError(
        f"Stage 2 managed vLLM port is unavailable at {host}:{port}: {last_error}"
    )


def _server_model_ids(endpoint: str, *, api_key: str, timeout: float) -> list[str]:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(f"{endpoint}/models", headers=headers)
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - configured local server
        payload = json.loads(response.read().decode("utf-8"))
    rows = payload.get("data") if isinstance(payload, Mapping) else None
    if not isinstance(rows, list):
        return []
    return sorted(
        {
            str(row.get("id") or "").strip()
            for row in rows
            if isinstance(row, Mapping) and str(row.get("id") or "").strip()
        }
    )


@dataclass
class _ServerProcess:
    index: int
    endpoint: str
    port: int
    internal_port_base: int
    logical_gpus: tuple[str, ...]
    visible_gpu_tokens: tuple[str, ...]
    command: list[str]
    log_path: Path
    log_handle: Any
    process: subprocess.Popen[bytes]
    model_ids: tuple[str, ...] = ()


class ManagedVLLMServerPool:
    """Start, health-check, and stop a fixed set of local vLLM replicas."""

    def __init__(
        self,
        *,
        config: ManagedVLLMConfig,
        model: str,
        api_key: str,
        output_dir: Path,
    ) -> None:
        config.validate()
        if not str(model).strip():
            raise ValueError("a model is required for pipeline-managed vLLM servers")
        self.config = config
        self.model = str(model).strip()
        self.api_key = str(api_key)
        self.output_dir = Path(output_dir)
        self.manifest_path = self.output_dir / "manifest.json"
        self._servers: list[_ServerProcess] = []
        self._stopped = False

    @property
    def endpoints(self) -> tuple[str, ...]:
        return tuple(server.endpoint for server in self._servers)

    def _manifest(self, *, status: str, error: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": "stage2_managed_vllm_pool_v1",
            "status": status,
            "updated_at": _now(),
            "model": self.model,
            "config": self.config.public_dict(),
            "servers": [
                {
                    "index": server.index,
                    "pid": server.process.pid,
                    "endpoint": server.endpoint,
                    "port": server.port,
                    "vllm_internal_port_base": server.internal_port_base,
                    "logical_gpus": list(server.logical_gpus),
                    "cuda_visible_devices": list(server.visible_gpu_tokens),
                    "tensor_parallel_size": len(server.logical_gpus),
                    "command": _redacted_command(server.command),
                    "log_path": str(server.log_path),
                    "model_ids": list(server.model_ids),
                    "exit_code": server.process.poll(),
                }
                for server in self._servers
            ],
        }
        if error:
            payload["error"] = error
        return payload

    def _log_tail(self, server: _ServerProcess, *, lines: int = 40) -> str:
        try:
            values = server.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return ""
        return "\n".join(values[-lines:])

    def start(self) -> tuple[str, ...]:
        if self._servers:
            raise RuntimeError("managed vLLM server pool has already been started")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        ports = self.config.effective_ports()
        gpu_groups = self.config.gpu_groups()
        internal_port_bases = _vllm_internal_port_bases(
            self.config.server_count,
            internal_port_base=self.config.internal_port_base,
        )
        parent_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        for port in ports:
            _assert_port_available(self.config.host, port)

        try:
            for index, (port, logical_gpus, internal_port_base) in enumerate(
                zip(ports, gpu_groups, internal_port_bases, strict=True)
            ):
                visible_tokens = _visible_gpu_tokens(
                    logical_gpus,
                    parent_cuda_visible_devices=parent_visible,
                )
                command = _build_vllm_command(
                    self.config,
                    model=self.model,
                    api_key=self.api_key,
                    port=port,
                    tensor_parallel_size=len(logical_gpus),
                )
                log_path = self.output_dir / f"server_{index:03d}.log"
                log_handle = log_path.open("ab", buffering=0)
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = ",".join(visible_tokens)
                # OCI_PYTHON selects the interpreter without activating its
                # environment.  Prepending its bin directory makes companion
                # tools installed there (notably ninja for FlashInfer JIT)
                # available to the vLLM child process.
                interpreter_bin = str(Path(sys.executable).parent)
                existing_path = environment.get("PATH", "")
                environment["PATH"] = (
                    interpreter_bin
                    if not existing_path
                    else os.pathsep.join((interpreter_bin, existing_path))
                )
                environment["VLLM_PORT"] = str(internal_port_base)
                environment.setdefault("PYTHONUNBUFFERED", "1")
                try:
                    process = subprocess.Popen(
                        command,
                        env=environment,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                except BaseException:
                    log_handle.close()
                    raise
                server = _ServerProcess(
                    index=index,
                    endpoint=_endpoint(self.config.host, port),
                    port=port,
                    internal_port_base=internal_port_base,
                    logical_gpus=tuple(logical_gpus),
                    visible_gpu_tokens=visible_tokens,
                    command=command,
                    log_path=log_path,
                    log_handle=log_handle,
                    process=process,
                )
                self._servers.append(server)
                LOGGER.info(
                    "started managed vLLM server=%s pid=%s endpoint=%s logical_gpus=%s log=%s",
                    index,
                    process.pid,
                    server.endpoint,
                    list(logical_gpus),
                    log_path,
                )
            _write_json(self.manifest_path, self._manifest(status="starting"))
            self._wait_until_ready()
            _write_json(self.manifest_path, self._manifest(status="ready"))
            return self.endpoints
        except BaseException as exc:
            if self._stopped:
                raise
            error = f"{type(exc).__name__}: {exc}"
            if self._servers:
                _write_json(
                    self.manifest_path,
                    self._manifest(status="failed", error=error),
                )
            self.stop(final_status="failed", error=error)
            raise

    def _wait_until_ready(self) -> None:
        pending = {server.index for server in self._servers}
        deadline = time.monotonic() + self.config.startup_timeout
        last_errors: dict[int, str] = {}
        while pending:
            for server in self._servers:
                if server.index not in pending:
                    continue
                exit_code = server.process.poll()
                if exit_code is not None:
                    tail = self._log_tail(server)
                    raise RuntimeError(
                        f"managed vLLM server {server.index} exited with code {exit_code} "
                        f"before readiness; log={server.log_path}\n{tail}"
                    )
                try:
                    model_ids = _server_model_ids(
                        server.endpoint,
                        api_key=self.api_key,
                        timeout=min(5.0, self.config.startup_poll_interval),
                    )
                    if model_ids:
                        server.model_ids = tuple(model_ids)
                        pending.remove(server.index)
                        LOGGER.info(
                            "managed vLLM server ready server=%s endpoint=%s models=%s",
                            server.index,
                            server.endpoint,
                            model_ids,
                        )
                    else:
                        last_errors[server.index] = "models endpoint advertised no models"
                except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
                    last_errors[server.index] = f"{type(exc).__name__}: {exc}"
            if not pending:
                return
            if time.monotonic() >= deadline:
                details = "; ".join(
                    f"server {index}: {last_errors.get(index, 'not ready')}"
                    for index in sorted(pending)
                )
                raise TimeoutError(
                    "timed out waiting for managed Stage 2 vLLM servers after "
                    f"{self.config.startup_timeout:g}s ({details})"
                )
            time.sleep(min(self.config.startup_poll_interval, 5.0))

    @staticmethod
    def _signal_process_group(server: _ServerProcess, sig: signal.Signals) -> None:
        if server.process.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(server.process.pid), sig)
        except ProcessLookupError:
            return
        except (AttributeError, OSError):
            if sig == signal.SIGTERM:
                server.process.terminate()
            else:
                server.process.kill()

    def stop(
        self,
        *,
        final_status: str = "stopped",
        error: str | None = None,
    ) -> None:
        if not self._servers or self._stopped:
            return
        for server in self._servers:
            self._signal_process_group(server, signal.SIGTERM)
        deadline = time.monotonic() + self.config.shutdown_timeout
        while any(server.process.poll() is None for server in self._servers):
            if time.monotonic() >= deadline:
                for server in self._servers:
                    self._signal_process_group(server, signal.SIGKILL)
                break
            time.sleep(0.1)
        for server in self._servers:
            try:
                server.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._signal_process_group(server, signal.SIGKILL)
                server.process.wait(timeout=5.0)
            server.log_handle.close()
            LOGGER.info(
                "stopped managed vLLM server=%s pid=%s exit_code=%s",
                server.index,
                server.process.pid,
                server.process.returncode,
            )
        _write_json(
            self.manifest_path,
            self._manifest(status=final_status, error=error),
        )
        self._stopped = True


@contextmanager
def launch_managed_vllm_servers(
    *,
    config: ManagedVLLMConfig,
    model: str,
    api_key: str,
    output_dir: Path,
) -> Iterator[tuple[str, ...]]:
    pool = ManagedVLLMServerPool(
        config=config,
        model=model,
        api_key=api_key,
        output_dir=output_dir,
    )
    previous_handlers: dict[signal.Signals, Any] = {}

    def stop_for_signal(signum: int, frame: Any) -> None:
        received = signal.Signals(signum)
        LOGGER.warning("stopping managed vLLM servers after signal=%s", received.name)
        try:
            pool.stop(final_status="interrupted")
        except Exception:
            LOGGER.exception("managed vLLM cleanup failed while handling %s", received.name)
        previous = previous_handlers.get(received, signal.SIG_DFL)
        if callable(previous):
            previous(signum, frame)
        raise SystemExit(128 + int(signum))

    if threading.current_thread() is threading.main_thread():
        managed_signals = [signal.SIGINT, signal.SIGTERM]
        if hasattr(signal, "SIGHUP"):
            managed_signals.append(signal.SIGHUP)
        for managed_signal in managed_signals:
            previous = signal.getsignal(managed_signal)
            if previous == signal.SIG_IGN:
                continue
            previous_handlers[managed_signal] = previous
            signal.signal(managed_signal, stop_for_signal)
    try:
        endpoints = pool.start()
        yield endpoints
    finally:
        try:
            pool.stop()
        finally:
            for managed_signal, previous in previous_handlers.items():
                if signal.getsignal(managed_signal) is stop_for_signal:
                    signal.signal(managed_signal, previous)


__all__ = [
    "ManagedVLLMConfig",
    "ManagedVLLMServerPool",
    "launch_managed_vllm_servers",
    "managed_vllm_config_from_mapping",
    "validate_managed_vllm_pool_isolation",
]
