"""Live, client-side admission limits for a Stage 2 run sharing one server."""

from __future__ import annotations

from contextlib import contextmanager
import json
import logging
from pathlib import Path
import threading
import time
from typing import Iterator, Mapping

LOGGER = logging.getLogger(__name__)


class LiveRequestLimits:
    """Reload small JSON limits while letting admitted requests finish normally.

    The file contains interpretation, extraction, and total nonnegative integer
    limits. Zero pauses new admissions. Invalid edits retain the last valid
    limits. This gate must wrap logical requests before their deadline starts.
    """

    def __init__(
        self, path: Path, *, ceilings: Mapping[str, int], poll_seconds: float = 1.0
    ) -> None:
        self.path = Path(path)
        self.ceilings = dict(ceilings)
        if set(self.ceilings) != {"interpretation", "extraction", "total"}:
            raise ValueError("Concurrency ceilings require both roles and total")
        if any(type(value) is not int or value < 1 for value in self.ceilings.values()):
            raise ValueError("Concurrency ceilings must be positive integers")
        if poll_seconds <= 0:
            raise ValueError("Concurrency poll interval must be positive")
        self.poll_seconds = float(poll_seconds)
        self._limits = self._validate(json.loads(self.path.read_text()))
        self._active = {"interpretation": 0, "extraction": 0}
        self._condition = threading.Condition()
        self._next_read = 0.0
        self._last_error = None
        LOGGER.info("Stage 2 live concurrency limits=%s path=%s", self._limits, self.path)

    def _validate(self, raw: Mapping[str, int]) -> dict[str, int]:
        if not isinstance(raw, dict) or set(raw) != set(self.ceilings):
            raise ValueError("Concurrency policy requires interpretation, extraction, total")
        if any(
            type(raw[key]) is not int or not 0 <= raw[key] <= ceiling
            for key, ceiling in self.ceilings.items()
        ):
            raise ValueError(f"Concurrency limits must be integers within {self.ceilings}")
        return dict(raw)

    def _reload_locked(self) -> None:
        now = time.monotonic()
        if now < self._next_read:
            return
        self._next_read = now + self.poll_seconds
        try:
            limits = self._validate(json.loads(self.path.read_text()))
        except (OSError, ValueError) as exc:
            error = str(exc)
            if error != self._last_error:
                LOGGER.warning("Keeping last valid Stage 2 concurrency limits: %s", error)
            self._last_error = error
            return
        self._last_error = None
        if limits != self._limits:
            LOGGER.info(
                "Stage 2 live concurrency changed old=%s new=%s active=%s",
                self._limits, limits, self._active,
            )
            self._limits = limits
            self._condition.notify_all()

    @contextmanager
    def slot(self, role: str) -> Iterator[None]:
        if role not in self._active:
            raise ValueError(f"Unknown Stage 2 request role: {role}")
        with self._condition:
            while True:
                self._reload_locked()
                if (
                    self._active[role] < self._limits[role]
                    and sum(self._active.values()) < self._limits["total"]
                ):
                    self._active[role] += 1
                    break
                self._condition.wait(timeout=self.poll_seconds)
        try:
            yield
        finally:
            with self._condition:
                self._active[role] -= 1
                self._condition.notify_all()
