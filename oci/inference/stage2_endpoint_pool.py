"""Capacity and load tracking for equivalent, externally hosted extractors."""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
import threading
import time
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlparse
from urllib.request import Request, urlopen

@dataclass(frozen=True)
class ExtractionEndpoint:
    endpoint: str
    max_concurrency: int

    def __post_init__(self) -> None:
        if isinstance(self.endpoint, str):
            object.__setattr__(self, "endpoint", self.endpoint.strip().rstrip("/"))

    def validate(self) -> None:
        if not isinstance(self.endpoint, str):
            raise ValueError("extraction endpoint must be a URL string")
        parsed = urlparse(self.endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("extraction endpoints must be HTTP(S) OpenAI-compatible base URLs")
        try:
            parsed.port
        except ValueError as error:
            raise ValueError("extraction endpoint has an invalid port") from error
        if type(self.max_concurrency) is not int or self.max_concurrency < 1:
            raise ValueError("extraction endpoint max_concurrency must be a positive integer")

    def public_dict(self) -> dict[str, Any]:
        return {"endpoint": self.endpoint, "max_concurrency": self.max_concurrency}


def endpoints_from_mapping(value: Any, *, default_concurrency: int) -> tuple[ExtractionEndpoint, ...]:
    # Resolved dataclass snapshots include an empty tuple/list for legacy single
    # routes. It means no pool; the caller still requires a valid selected route.
    if value is None or isinstance(value, (list, tuple)) and not value:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError("stage2.extraction_llm.endpoints must be a list")
    endpoints = []
    for index, entry in enumerate(value):
        if isinstance(entry, str):
            endpoint, concurrency = entry, default_concurrency
        elif isinstance(entry, Mapping):
            unknown = set(entry) - {"endpoint", "max_concurrency"}
            if unknown or "endpoint" not in entry or "max_concurrency" not in entry:
                raise ValueError(
                    f"stage2.extraction_llm.endpoints[{index}] requires endpoint and max_concurrency"
                )
            endpoint, concurrency = entry["endpoint"], entry["max_concurrency"]
        else:
            raise ValueError(f"stage2.extraction_llm.endpoints[{index}] must be a URL or object")
        if not isinstance(endpoint, str):
            raise ValueError(f"stage2.extraction_llm.endpoints[{index}].endpoint must be a URL")
        item = ExtractionEndpoint(endpoint, concurrency)
        item.validate()
        endpoints.append(item)
    urls = [item.endpoint for item in endpoints]
    if len(urls) != len(set(urls)):
        raise ValueError("stage2.extraction_llm.endpoints contains duplicate URLs")
    return tuple(endpoints)


def read_vllm_load(endpoint: str, api_key: str) -> dict[str, float]:
    """Read aggregate queue counts and the busiest engine's KV-cache fraction."""
    url = endpoint.rstrip("/").removesuffix("/v1") + "/metrics"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key and api_key != "EMPTY" else {}
    with urlopen(Request(url, headers=headers), timeout=2) as response:
        raw = response.read(2 * 1024 * 1024 + 1)
    if len(raw) > 2 * 1024 * 1024:
        raise ValueError("vLLM metrics response exceeds the load-monitor size limit")
    wanted = {
        "vllm:num_requests_running": "running",
        "vllm:num_requests_waiting": "waiting",
        "vllm:kv_cache_usage_perc": "cache_fraction",
    }
    values: dict[str, list[float]] = {}
    for line in raw.decode().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Label values may contain spaces; the numeric sample follows the final
        # closing brace, rather than necessarily being the second word.
        if "{" in line:
            name = line.split("{", 1)[0]
            samples = line.rsplit("}", 1)[-1].split()
        else:
            name, *samples = line.split()
        if name not in wanted or not samples:
            continue
        try:
            sample = float(samples[0])
        except ValueError:
            continue
        if math.isfinite(sample) and sample >= 0:
            values.setdefault(wanted[name], []).append(sample)
    result = {key: max(samples) if key == "cache_fraction" else sum(samples)
              for key, samples in values.items()}
    if not result:
        raise ValueError("endpoint does not expose recognized vLLM load metrics")
    return result


@dataclass
class _ServerState:
    config: ExtractionEndpoint
    in_flight: int = 0
    latency_seconds: float | None = None
    consecutive_failures: int = 0
    cooldown_until: float = 0.0
    metrics: dict[str, float] | None = None
    metrics_at: float = -math.inf
    local_at_metrics: int = 0
    next_metrics_poll: float = 0.0
    metrics_pending: bool = False


class EndpointPool:
    """Reserve server capacity using local load, latency, and optional shared load.

    Metrics refresh in short-lived background threads; network monitoring never
    blocks an extraction request. Missing/stale metrics fall back to local load.
    HTTP transport failures impose bounded cooldowns with one recovery probe.
    Homogeneous decision replicas can ignore historical response latency so
    slow startup samples do not permanently exclude otherwise idle capacity.
    """

    def __init__(
        self, endpoints: Sequence[ExtractionEndpoint], *, api_key: str = "EMPTY",
        metrics_reader: Callable[[str, str], Mapping[str, float]] | None = read_vllm_load,
        clock: Callable[[], float] = time.monotonic,
        metrics_poll_seconds: float = 5.0,
        latency_weighted: bool = True,
    ) -> None:
        if not endpoints or metrics_poll_seconds <= 0:
            raise ValueError("endpoint pool requires servers and a positive metrics interval")
        for endpoint in endpoints:
            endpoint.validate()
        urls = [endpoint.endpoint for endpoint in endpoints]
        if len(set(urls)) != len(urls):
            raise ValueError("endpoint pool contains duplicate URLs")
        self._states = [_ServerState(endpoint) for endpoint in endpoints]
        self.capacity = sum(endpoint.max_concurrency for endpoint in endpoints)
        self._api_key, self._metrics_reader, self._clock = api_key, metrics_reader, clock
        self._metrics_poll_seconds = metrics_poll_seconds
        self._latency_weighted = latency_weighted
        self._condition = threading.Condition()
        self._cursor = 0

    def _poll_metrics(self, index: int, local_at_poll: int) -> None:
        state = self._states[index]
        try:
            assert self._metrics_reader is not None
            raw = self._metrics_reader(state.config.endpoint, self._api_key)
            metrics = {key: float(value) for key, value in raw.items()
                       if key in {"running", "waiting", "cache_fraction"}
                       and not isinstance(value, bool)
                       and math.isfinite(float(value)) and float(value) >= 0}
            if not metrics:
                raise ValueError("no usable load metrics")
        except Exception:
            # A /metrics outage does not establish that the inference API is down.
            metrics = None
        with self._condition:
            if metrics is not None:
                state.metrics, state.metrics_at = metrics, self._clock()
                state.local_at_metrics = local_at_poll
            state.metrics_pending = False
            self._condition.notify_all()

    def _refresh_metrics(self) -> None:
        if self._metrics_reader is None:
            return
        jobs = []
        with self._condition:
            now = self._clock()
            for index, state in enumerate(self._states):
                if not state.metrics_pending and now >= state.next_metrics_poll:
                    state.metrics_pending = True
                    state.next_metrics_poll = now + self._metrics_poll_seconds
                    jobs.append((index, state.in_flight))
        for index, local_at_poll in jobs:
            threading.Thread(target=self._poll_metrics, args=(index, local_at_poll),
                             name=f"stage2-server-load-{index}", daemon=True).start()

    def _score(self, state: _ServerState, now: float, reference_latency: float) -> float:
        external, pressure = 0.0, 1.0
        if state.metrics is not None and now - state.metrics_at <= 3 * self._metrics_poll_seconds:
            reported = state.metrics.get("running", 0) + state.metrics.get("waiting", 0)
            external = max(0.0, reported - state.local_at_metrics)
            cache = min(1.0, state.metrics.get("cache_fraction", 0))
            pressure += 4 * max(0.0, cache - 0.8) / 0.2
        latency = (state.latency_seconds or reference_latency) if self._latency_weighted else 1.0
        return (state.in_flight + external + 1) / state.config.max_concurrency * latency * pressure

    def reserve(self, *, deadline: float | None = None) -> int:
        self._refresh_metrics()
        with self._condition:
            while True:
                now = self._clock()
                if deadline is not None and now >= deadline:
                    raise TimeoutError("logical request deadline expired waiting for extractor capacity")
                latencies = [state.latency_seconds for state in self._states if state.latency_seconds]
                reference = statistics.median(latencies) if latencies else 1.0
                candidates = [index for index, state in enumerate(self._states)
                              if state.in_flight < state.config.max_concurrency
                              and now >= state.cooldown_until
                              and not (state.consecutive_failures and state.in_flight)]
                if candidates:
                    index = min(candidates, key=lambda item: (
                        self._score(self._states[item], now, reference),
                        (item - self._cursor) % len(self._states)))
                    self._states[index].in_flight += 1
                    self._cursor = (index + 1) % len(self._states)
                    return index
                waits = [1.0]
                waits.extend(state.cooldown_until - now for state in self._states
                             if state.cooldown_until > now)
                if deadline is not None:
                    waits.append(deadline - now)
                self._condition.wait(timeout=max(0.001, min(waits)))

    def release(self, index: int, *, duration_seconds: float, transport_failed: bool = False,
                succeeded: bool = True) -> None:
        with self._condition:
            state = self._states[index]
            state.in_flight -= 1
            if transport_failed:
                state.consecutive_failures += 1
                state.cooldown_until = self._clock() + min(30.0, 2 ** min(state.consecutive_failures, 5))
            elif succeeded:
                state.consecutive_failures, state.cooldown_until = 0, 0.0
                duration = max(0.001, duration_seconds)
                state.latency_seconds = (duration if state.latency_seconds is None else
                                         0.8 * state.latency_seconds + 0.2 * duration)
            self._condition.notify_all()

    def server(self, index: int) -> ExtractionEndpoint:
        return self._states[index].config

    def snapshot(self) -> list[dict[str, Any]]:
        with self._condition:
            now = self._clock()
            return [dict(**state.config.public_dict(), in_flight=state.in_flight,
                         latency_seconds=state.latency_seconds,
                         cooldown_seconds=max(0, state.cooldown_until - now),
                         metrics=state.metrics if now-state.metrics_at <= 3*self._metrics_poll_seconds else None)
                    for state in self._states]
