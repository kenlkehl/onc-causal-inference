"""Reproduce one real frozen query's probes without reading validation labels."""
import os
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
import numpy as np
import pandas as pd
from common import RUN, POLICY, labels, nuisance_path, read, split, write, now, verify_manifest
from probes import Probe


def main():
    verify_manifest()
    part = split()["inner_splits"][0]
    folder = RUN / "folds/inner_001"
    labels_train = labels(part["fit_row_ids"])
    t, y = labels_train.treatment.to_numpy(), labels_train.outcome.to_numpy()
    nuisance = pd.read_parquet(nuisance_path(1)).set_index("_oci_row_id")
    e, m = nuisance.loc[part["fit_row_ids"], ["treatment_stacked", "outcome_stacked"]].to_numpy().T
    u, v = t-e, y-m
    arrays = np.load(folder / "activations.npz", allow_pickle=False)
    frozen = np.load(folder / "probe_predictions.npz", allow_pickle=False)
    metadata = read(folder / "probe_metadata.json")
    variants = read(folder / "variants.json")
    key = lambda d: (d["query"], d["scope"], d["family"], d["target"], d["mode"], d["variant"])
    lookup = {key(d): index for index, d in enumerate(metadata)}
    index = next(i for i, x in enumerate(variants) if x["query"] == 0 and x["kind"] == "targeted" and x["radius"] == 0.2)
    errors = []
    for scope in ("solo", "conditional"):
        x0 = arrays["original_train"][:, [10]] if scope == "solo" else arrays["original_train"]
        h0 = arrays["original_validation"][:, [10]] if scope == "solo" else arrays["original_validation"]
        x1 = arrays["train"][:, [index]] if scope == "solo" else x0.copy()
        h1 = arrays["validation"][:, [index]] if scope == "solo" else h0.copy()
        if scope == "conditional":
            x1[:, 10], h1[:, 10] = arrays["train"][:, index], arrays["validation"][:, index]
        for family in ("linear", "spline"):
            for target in ("treatment", "outcome", "effect"):
                model = Probe(family, target, POLICY).fit(x0, t, y, u, v)
                refit = Probe(family, target, POLICY).fit(x1, t, y, u, v)
                for mode, variant, prediction in (("original", 0, model.predict(h0)),
                                                  ("frozen", index, model.predict(h1)),
                                                  ("refit", index, refit.predict(h1))):
                    expected = frozen["predictions"][lookup[(0, scope, family, target, mode, variant)]]
                    error = float(np.max(np.abs(prediction - expected)))
                    assert error < 1e-8, (scope, family, target, mode, error)
                    errors.append(error)
    write(RUN / "REPLAY_VALIDATION_2026-09-22.json", {"at": now(), "predictions_reproduced": len(errors),
          "maximum_absolute_error": max(errors), "inner_validation_labels_loaded": False,
          "outer_test_rows_used": 0, "original_frozen_and_refit_modes_checked": True})
    print(read(RUN / "REPLAY_VALIDATION_2026-09-22.json"))


if __name__ == "__main__":
    main()
