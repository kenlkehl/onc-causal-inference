"""Shared provenance helpers for the training-only consolidation experiment."""
from pathlib import Path
import importlib.util

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "fold_1_multi_model_selection_2026-09-21"
spec = importlib.util.spec_from_file_location("original_numerical_run", BASE / "numerical_2026-09-21.py")
n = importlib.util.module_from_spec(spec)
spec.loader.exec_module(n)
DATE = n.DATE
PREFIX = "outer_001_feature_"
FAMILIES = {"U": "univariable", "PM": "penalized_main", "PI": "penalized_interactions",
            "OL": "orthogonal_linear", "UR": "univariable_rlearner",
            "PF": "predictive_forest", "CF": "causal_forest"}
ROLES = {"T": "treatment", "Y": "outcome", "E": "effect"}


def full_id(short):
    return PREFIX + short


def evidence_id(short, reference):
    family, role = reference.split(":")
    return f"multi:{full_id(short)}:{FAMILIES[family]}:{ROLES[role]}"


def assert_pre_oracle():
    assert not (BASE / "oracle_access_started.json").exists()
    assert not (HERE / "oracle_access_started.json").exists()


def verify_upstream():
    manifest = n.read(BASE / f"input_manifest_{DATE}.json")
    n.verify(manifest["sources"])
    n.verify(manifest["input_files"])
    n.verify(n.read(BASE / f"numerical_frozen_{DATE}.json")["files"])
    n.verify(n.read(BASE / f"selection_frozen_{DATE}.json")["files"])
    return manifest
