"""Run all seven Stage 2 evidence families on frozen outer-fold-1 measurements.

Reads only observed training treatment/outcome and extracted candidate values.
Does not load held-out labels, oracle variables, or the synthetic source dataset.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/stage2-mpl")
INPUTS = HERE.parent / "fold_1_extracted_logistic_interactions_2026-09-21"
SOURCE = ROOT / "reports/2026-09-19/selection_comparison_2026-09-19/results"
DATE = "2026-09-21"
SEED = 100042


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def verify(files):
    for path, expected in files.items():
        if sha(path) != expected:
            raise ValueError(f"Frozen file changed: {path}")


def main():
    import numpy as np
    import pandas as pd
    from oci.inference.stage2_elastic_net_selection import (
        statistical_selection_config_from_mapping,
        select_stage2_features_elastic_net,
    )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    frozen_inputs = read(INPUTS / f"inputs_frozen_{DATE}.json")
    verify(frozen_inputs["files"])
    definitions = read(INPUTS / "inputs/definitions.json")
    split = read(INPUTS / "inputs/split.json")
    training = pd.read_pickle(INPUTS / "inputs/training.pkl")
    labels = pd.read_parquet(INPUTS / "inputs/training_labels.parquet",
                             columns=["_oci_row_id", "treatment", "outcome"])
    assert labels._oci_row_id.tolist() == training._oci_row_id.tolist() == split["fit_row_ids"]
    assert len(training) == 800 and len(definitions) == 352
    # Only training labels are populated; outer-test rows cannot enter selection.
    dataset = pd.DataFrame(np.nan, index=range(1000), columns=["treatment", "outcome"])
    dataset.loc[labels._oci_row_id, ["treatment", "outcome"]] = labels[["treatment", "outcome"]].to_numpy()
    original_path = SOURCE / "refresh/outer_001/selection/input.json"
    original = read(original_path)
    assert definitions == original["definitions"]
    assert split["inner_splits"] == original["inner_splits"]
    config = dict(original["statistical_selection_policy"], selection_mode="multi_model")
    policy = statistical_selection_config_from_mapping(config)
    source_paths = [Path(__file__), original_path, INPUTS / f"inputs_frozen_{DATE}.json"]
    source_paths += [ROOT / rel for rel in (
        "oci/inference/stage2_multi_model_selection.py",
        "oci/inference/stage2_multi_model_config.py",
        "oci/inference/stage2_multi_model_adjudication.py",
        "oci/inference/stage2_role_adjudication.py",
        "oci/inference/stage2_elastic_net_selection.py",
        "oci/inference/plain_handoff_stage2.py",
        "oci/inference/plain_handoff_stage2_analysis.py",
        "oci/models/causal_forest_head.py",
        "oci/models/elastic_net_nuisance.py",
    )]
    manifest_path = HERE / f"input_manifest_{DATE}.json"
    manifest = {
        "created_at": now(), "outer_fold": 1, "seed": SEED,
        "training_rows": 800, "heldout_rows": 200, "candidate_features": 352,
        "policy": policy.public_dict(),
        "role_policy": dict(original["role_adjudication_policy"], enabled=True),
        "adjudication": {"endpoint": "http://sn4622130540:8000/v1", "model": "gemma4-31b",
                         "reasoning_effort": "high", "max_tokens": 100000},
        "evaluation_plan": {
            "production": "Existing estimate_outer_fold; selected confounders for nuisance adjustment and modifiers for X; three seeds.",
            "estimation_seeds": [100042, 1100042, 2100042],
            "trees": 200, "min_propensity": 0.1, "max_propensity": 0.9, "propensity_clip": 0.02,
            "matched_selection_comparison": "Also use frozen original all-candidate elastic-net residuals and overlap population (720/180) with selected modifiers in X, historical sqrt forest, three seeds.",
            "matched_forest_seeds": [120042, 1120042, 2120042],
            "oracle_recovery_and_ite_evaluation": "Only after selected definitions and all new predictions have been frozen.",
        },
        "prior_fold_1_oracle_results_seen": True,
        "oracle_used_by_selection_or_fitting": False,
        "outer_heldout_used_in_selection": False, "extraction_refreshed_again": False,
        "input_files": frozen_inputs["files"],
        "sources": {str(p.resolve()): sha(p) for p in source_paths},
        "versions": {p: importlib.metadata.version(p) for p in
                     ["numpy", "pandas", "scipy", "scikit-learn", "econml", "statsmodels"]},
    }
    if manifest_path.exists():
        previous = read(manifest_path)
        manifest["created_at"] = previous["created_at"]
        assert previous == manifest, "Numerical input manifest changed"
    else:
        write(manifest_path, manifest)
    write(HERE / "status.json", {"phase": "numerical_evidence", "updated_at": now(), "pid": os.getpid()})
    _, report, returned, latent = select_stage2_features_elastic_net(
        dataset=dataset, extracted_fit=training, definitions=definitions,
        inner_splits=split["inner_splits"], treatment_column="treatment", outcome_column="outcome",
        outcome_type="binary", seed=SEED, policy=policy, checkpoint_dir=HERE / "numerical",
    )
    assert returned == definitions and not latent
    verify(manifest["sources"])
    verify(manifest["input_files"])
    path = HERE / "statistical_evidence.json"
    write(path, report)
    write(HERE / f"numerical_frozen_{DATE}.json", {
        "frozen_at": now(), "input_manifest_sha256": sha(manifest_path),
        "files": {str(path.resolve()): sha(path)}, "evaluable_cells_by_family": report["evaluable_cells_by_family"],
        "input_fingerprint": report["input_fingerprint"],
    })
    write(HERE / "status.json", {"phase": "numerical_complete", "updated_at": now(),
                               "availability": report["evaluable_cells_by_family"]})
    print(json.dumps({"phase": "numerical_complete", "availability": report["evaluable_cells_by_family"]}), flush=True)


if __name__ == "__main__":
    main()
