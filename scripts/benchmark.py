#!/usr/bin/env python3
"""Benchmark harness: materialize scenario cohorts, write run configurations, aggregate.

``materialize`` builds one cohort per scenario, size and seed (re-equation, or
the generator commands for scenarios that need LLM text); ``configs`` writes one
``run_all_evidence`` configuration per cell and prints the commands; ``aggregate``
compares the finished runs with the oracle columns. It never launches a run or
an LLM request. See ``oci/evaluation/benchmark.py`` and ``docs/benchmarks.md``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from oci.evaluation.benchmark import main

if __name__ == "__main__":
    raise SystemExit(main())
