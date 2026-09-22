"""Immutable-input helpers for an explicitly post-hoc fold-1 comparison."""
from pathlib import Path
import importlib.util

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "fold_1_multi_model_selection_2026-09-21"
PRIOR = HERE.parent / "fold_1_multi_model_consolidation_2026-09-21"
spec = importlib.util.spec_from_file_location("original_numerical_run", BASE / "numerical_2026-09-21.py")
n = importlib.util.module_from_spec(spec)
spec.loader.exec_module(n)
DATE = "2026-09-21"


def verify_inputs():
    manifest = n.read(HERE / f"input_manifest_{DATE}.json")
    selection = n.read(HERE / f"selection_frozen_{DATE}.json")
    assert selection["input_manifest_sha256"] == n.sha(HERE / f"input_manifest_{DATE}.json")
    for files in (manifest["sources"], manifest["input_files"], manifest["upstream_files"], selection["files"]):
        n.verify(files)
    return manifest, selection
