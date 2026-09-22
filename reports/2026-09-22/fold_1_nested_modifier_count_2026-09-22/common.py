"""Frozen-input helpers for the fold-1 nested modifier-count experiment."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
DATE = "2026-09-22"
OLD_DATE = "2026-09-21"
BASE = ROOT / f"reports/{OLD_DATE}/fold_1_multi_model_selection_{OLD_DATE}"
PRIOR = ROOT / f"reports/{OLD_DATE}/fold_1_conservative_confounders_moderate_modifiers_{OLD_DATE}"
COMPACT = ROOT / f"reports/{OLD_DATE}/fold_1_multi_model_consolidation_{OLD_DATE}"
INPUTS = ROOT / f"reports/{OLD_DATE}/fold_1_extracted_logistic_interactions_{OLD_DATE}/inputs"
SOURCE = ROOT / "reports/2026-09-19/selection_comparison_2026-09-19/results"
SEED = 100042
MODEL = "gemma4-31b"
ENDPOINT = "http://sn4622130540:8000/v1"


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Threads use separate request files; other artifacts have one writer.
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def verify(files):
    for path, expected in files.items():
        if sha(path) != expected:
            raise ValueError(f"Frozen input changed: {path}")


def manifest():
    value = read(HERE / f"input_manifest_{DATE}.json")
    verify(value["input_files"])
    verify(value["sources"])
    return value


def load_training():
    import numpy as np
    import pandas as pd

    definitions = read(INPUTS / "definitions.json")
    split = read(INPUTS / "split.json")
    train = pd.read_pickle(INPUTS / "training.pkl")
    labels = pd.read_parquet(INPUTS / "training_labels.parquet", columns=["_oci_row_id", "treatment", "outcome"])
    assert labels._oci_row_id.tolist() == train._oci_row_id.tolist() == split["fit_row_ids"]
    assert len(train) == 800 and len(definitions) == 352
    data = pd.DataFrame(np.nan, index=range(1000), columns=["treatment", "outcome"])
    data.loc[labels._oci_row_id, ["treatment", "outcome"]] = labels[["treatment", "outcome"]].to_numpy()
    assert data.loc[split["heldout_row_ids"]].isna().all().all()
    return data, train, definitions, split


def policy_from_manifest():
    from oci.inference.stage2_elastic_net_selection import statistical_selection_config_from_mapping

    return statistical_selection_config_from_mapping(read(HERE / f"input_manifest_{DATE}.json")["policy"])


def numerical_arguments(position):
    """Reproduce the native selector's full and nested numerical jobs exactly."""
    import numpy as np
    import pandas as pd
    from oci.inference import stage2_multi_model_selection as numerical

    data, train, definitions, split = load_training()
    policy = policy_from_manifest()
    seed = SEED
    inner = split["inner_splits"]
    checkpoint = HERE / "numerical"
    if position:
        parent = inner[position - 1]
        ids = parent["fit_row_ids"]
        train = train.set_index("_oci_row_id", drop=False).loc[ids].reset_index(drop=True)
        nested = pd.DataFrame(np.nan, index=range(len(data)), columns=["treatment", "outcome"])
        nested.loc[ids] = data.loc[ids].to_numpy()
        data = nested
        seed = SEED + position * 10_000 + 700_000
        inner = [
            {"inner_fold": j, "fit_row_ids": np.asarray(ids)[fit].tolist(), "heldout_row_ids": np.asarray(ids)[hold].tolist()}
            for j, (fit, hold) in enumerate(numerical.linear._crossfit_indices(
                data.loc[ids, "treatment"].to_numpy(), requested_folds=policy.internal_cv_folds, seed=seed), 1)
        ]
        checkpoint = checkpoint / "modifier_count_training"
        assert data.drop(index=ids).isna().all().all()
        assert set(train._oci_row_id).isdisjoint(parent["heldout_row_ids"])
    return dict(dataset=data, extracted_fit=train, definitions=definitions, inner_splits=inner,
                treatment_column="treatment", outcome_column="outcome", outcome_type="binary",
                seed=seed, policy=policy, checkpoint_dir=checkpoint)
