#!/usr/bin/env python3
"""Prepare and compare leave-one-architecture-out Stage 2 ablations.

``prepare`` writes inputs for ``scripts/run_stage2_from_artifacts.py`` and
prints the commands; it never starts an LLM request. ``compare`` reads the
finished ablations. See ``oci/evaluation/ablation.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from oci.evaluation.ablation import main

if __name__ == "__main__":
    raise SystemExit(main())
