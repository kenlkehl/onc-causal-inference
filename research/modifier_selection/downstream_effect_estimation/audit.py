#!/usr/bin/env python3
"""Independently reconstruct every metric and summary from saved predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import logit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from core import prediction_metrics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    predictions = pd.read_parquet(args.output / "heldout_predictions.parquet")
    reported = pd.read_csv(args.output / "replications.csv")
    keys = ["mechanism", "n", "replicate", "modifier_input", "estimator"]
    rebuilt = []
    for values, group in predictions.groupby(keys, sort=True):
        row = dict(zip(keys, values))
        for scale, truth, prediction in (
            ("rd", group.truth_mu1 - group.truth_mu0, group.fitted_rd),
            (
                "log_or",
                logit(group.truth_mu1) - logit(group.truth_mu0),
                group.fitted_log_or,
            ),
        ):
            row.update({f"{scale}_{key}": value for key, value in prediction_metrics(truth, prediction).items()})
        rebuilt.append(row)
    rebuilt = pd.DataFrame(rebuilt)
    joined = reported.merge(rebuilt, on=keys, suffixes=("_reported", "_rebuilt"), validate="one_to_one")
    metrics = ["rd_pearson", "rd_rmse", "rd_r2", "rd_spearman", "log_or_pearson", "log_or_rmse", "log_or_r2", "log_or_spearman"]
    maximum_error = max(
        float(np.max(np.abs(joined[f"{metric}_reported"] - joined[f"{metric}_rebuilt"])))
        for metric in metrics
    )
    result = {
        "status": "PASS" if maximum_error < 1e-12 and len(joined) == len(reported) else "FAIL",
        "replication_rows": len(reported),
        "reconstructed_rows": len(joined),
        "maximum_metric_error": maximum_error,
    }
    (args.output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    if result["status"] != "PASS":
        raise SystemExit(json.dumps(result))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
