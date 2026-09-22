"""Verify exact cached replay on the real fold with every numerical fitter disabled."""
import contextlib
import importlib.util
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("numerical_run", HERE / "numerical_2026-09-21.py")
n = importlib.util.module_from_spec(spec)
spec.loader.exec_module(n)


def main():
    import numpy as np
    import pandas as pd
    from oci.inference import stage2_multi_model_selection as multi
    from oci.inference.stage2_elastic_net_selection import (
        statistical_selection_config_from_mapping, select_stage2_features_elastic_net,
    )
    from oci.inference.stage2_role_adjudication import _fingerprint

    manifest = n.read(HERE / f"input_manifest_{n.DATE}.json")
    frozen = n.read(HERE / f"numerical_frozen_{n.DATE}.json")
    n.verify(frozen["files"])
    n.verify(manifest["sources"])
    n.verify(manifest["input_files"])
    definitions = n.read(n.INPUTS / "inputs/definitions.json")
    split = n.read(n.INPUTS / "inputs/split.json")
    training = pd.read_pickle(n.INPUTS / "inputs/training.pkl")
    labels = pd.read_parquet(n.INPUTS / "inputs/training_labels.parquet",
                             columns=["_oci_row_id", "treatment", "outcome"])
    dataset = pd.DataFrame(np.nan, index=range(1000), columns=["treatment", "outcome"])
    dataset.loc[labels._oci_row_id, ["treatment", "outcome"]] = labels[["treatment", "outcome"]].to_numpy()
    policy = statistical_selection_config_from_mapping(manifest["policy"])
    with contextlib.ExitStack() as stack:
        for name in ["_nuisances", "_univariable", "_penalized_family", "_univariable_rlearner", "_forests"]:
            stack.enter_context(patch.object(multi, name, side_effect=AssertionError("Unexpected numerical refit")))
        _, report, _, _ = select_stage2_features_elastic_net(
            dataset=dataset, extracted_fit=training, definitions=definitions,
            inner_splits=split["inner_splits"], treatment_column="treatment", outcome_column="outcome",
            outcome_type="binary", seed=manifest["seed"], policy=policy, checkpoint_dir=HERE / "numerical",
        )
    assert _fingerprint(report) == _fingerprint(n.read(HERE / "statistical_evidence.json"))
    n.verify(frozen["files"])
    n.write(HERE / f"numerical_replay_validation_{n.DATE}.json", {
        "validated_at": n.now(), "status": "passed", "numerical_refits": 0,
        "exact_cached_report_reproduced": True, "oracle_or_heldout_labels_read": False,
        "source_sha256": n.sha(__file__), "report_fingerprint": _fingerprint(report),
    })
    print("Exact numerical report reproduced from checkpoints with all numerical fitters disabled", flush=True)


if __name__ == "__main__":
    main()
