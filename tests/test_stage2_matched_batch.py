"""Patient boundaries, batch evidence, and its path into clinical review."""
from dataclasses import replace
import json

import numpy as np
import pandas as pd

from oci.inference import stage2_matched_batch as batch
from oci.inference.stage2_multi_model_config import MatchedBatchConfig
from oci.inference.stage2_multi_model_selection import aggregate_evidence
from oci.inference.stage2_multi_model_adjudication import build_multi_model_role_evidence
from oci.inference.stage2_role_adjudication import Stage2RoleAdjudicationConfig
from oci.inference.stage2_modifier_concepts import _concept_input
from tests.test_stage2_multi_model import feature, small_policy


def policy(**changes):
    cfg = replace(MatchedBatchConfig(enabled=True, train_passes=15, validation_passes=10), **changes)
    return replace(small_policy(), multi_model=replace(small_policy().multi_model, matched_batch=cfg))


def cohort():
    rng = np.random.default_rng(421)
    n, train_n = 1200, 900
    x = rng.normal(size=(n, 2))
    t = np.tile([0, 1], n // 2)
    y = (t - .5) * 2 * x[:, 0] + rng.normal(scale=.05, size=n)
    frame = pd.DataFrame(dict(modifier=x[:, 0], noise=x[:, 1], constant=1.))
    frame.insert(0, "_oci_row_id", np.arange(n))
    defs = [feature(k) for k in ("modifier", "noise", "constant")]
    return dict(train=frame.iloc[:train_n], valid=frame.iloc[train_n:], definitions=defs,
        t=t[:train_n], y=y[:train_n], tv=t[train_n:], yv=y[train_n:],
        nuisance={"e": np.full(train_n, .5), "m": np.zeros(train_n),
                  "ev": np.full(n - train_n, .5), "mv": np.zeros(n - train_n)},
        policy=policy(), seed=9)


def test_signal_recovers_on_separate_patients_and_reaches_concept_prompt():
    args = cohort()
    result = batch.score_candidates(**args)
    assert result == batch.score_candidates(**args)
    rows = {r["feature_id"]: r for r in result["records"]}
    assert rows["modifier"]["score"] > .7
    assert rows["noise"]["score"] < .1
    assert rows["constant"]["status"] == "not_evaluable"
    assert result["audit"]["variants"]["filtered"]["training_batches"] < result["audit"]["variants"]["unfiltered"]["training_batches"]
    assert result["audit"]["variants"]["filtered"]["validation_batches"] == result["audit"]["variants"]["unfiltered"]["validation_batches"]
    cell = {"family": "matched_batch_contrast", "inner_fold": 1, **result}
    ids = [f["feature_id"] for f in args["definitions"]]
    summary = aggregate_evidence([cell], ids)
    assert summary["modifier"][0]["exposures"] == 1
    assert summary["modifier"][0]["batch_diagnostics"]["mean_training_patients"] <= len(args["train"])
    summary["modifier"][0]["batch_diagnostics"]["patient_ids"] = "DO_NOT_LEAK"
    evidence = build_multi_model_role_evidence(definitions=args["definitions"],
        statistical_report={"multi_model_evidence": summary, "policy": args["policy"].public_dict()},
        policy=Stage2RoleAdjudicationConfig())
    text = _concept_input(evidence, evidence["candidates"], dict.fromkeys(ids, 1), 1)
    assert "DO_NOT_LEAK" not in text
    assert "10% error reduction" in text and "Shuffles reuse patients" in text
    assert "Filtered and unfiltered" in text
    assert "patient_ids" not in json.dumps(evidence)
