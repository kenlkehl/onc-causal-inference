from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path


def test_large_selector_runs_from_python_c_without_main_guard():
    repository = Path(__file__).resolve().parents[1]
    script = textwrap.dedent(
        """
        import os
        import oci.inference.plain_handoff_stage2_analysis as stage2_analysis

        parent_pid = os.getpid()
        stage2_analysis.select_stage2_features_multi_model = lambda **arguments: (
            arguments["definitions"][:1],
            {"worker_pid": os.getpid()},
            [],
            [],
        )
        result = stage2_analysis._run_stage2_statistical_selection(
            {
                "definitions": [
                    {"feature_id": f"feature_{index}"}
                    for index in range(64)
                ]
            }
        )
        assert result[0] == [{"feature_id": "feature_0"}]
        assert result[1]["worker_pid"] != parent_pid
        """
    )
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
