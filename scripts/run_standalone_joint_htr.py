#!/usr/bin/env python3
"""Standalone joint HTR (propensity, outcome, and tau in one network); see ``--help``."""

import sys
from pathlib import Path

# Use the checkout that contains this script, not another installed copy.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oci.inference.standalone_joint_htr import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
