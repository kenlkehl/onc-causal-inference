"""Scientific boundaries and end-to-end behavior of the multi-model selector."""


import json

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

from oci.inference.stage2_elastic_net_selection import (
    Stage2ElasticNetSelectionConfig,
)
from oci.inference.stage2_multi_model_config import (
    EvidenceStabilityConfig,
    Stage2MultiModelConfig,
)
from oci.inference.stage2_multi_model_selection import (
    aggregate_evidence,
)
from oci.inference.stage2_multi_model_adjudication import build_multi_model_role_evidence
from oci.inference.stage2_role_adjudication import (
    Stage2RoleAdjudicationConfig,
)


def feature(key):
    return {
        "feature_id": key,
        "name": key,
        "value_type": "continuous",
        "categories_or_unit": ["unitless"],
        "measurement_definition": "A measured baseline attribute.",
        "description": "A baseline measurement.",
        "ground_truth": "DO_NOT_LEAK",
        "source_dataset_path": "DO_NOT_LEAK",
    }


def small_policy():
    return Stage2ElasticNetSelectionConfig(
        internal_cv_folds=2,
        max_iter=1500,
        minimum_log10_alpha=-1.5,
        maximum_log10_alpha=0,
        optimization_tolerance=1e-4,
        categorical_min_count=2,
        multi_model=Stage2MultiModelConfig(
            stability=EvidenceStabilityConfig(partition_repeats=1, subset_partitions=1),
            repeats=2,
            regularization_grid_size=3,
            forest_trees=12,
            forest_min_samples_leaf=2,
            feature_subset_size=2,
            permutation_repeats=1,
        ),
    )


def sample_inputs():
    rng = np.random.default_rng(812)
    n = 120
    x = rng.normal(size=(n, 4))
    t = rng.binomial(1, 1 / (1 + np.exp(-x[:, 0])))
    y = x[:, 0] + t * (1.0 + 2.0 * (x[:, 1] > 0)) + rng.normal(scale=0.4, size=n)
    dataset = pd.DataFrame({"treatment": t, "outcome": y, "oracle": ["DO_NOT_READ"] * n})
    extracted = pd.DataFrame(x[:96], columns=["c", "m", "noise", "other"])
    extracted.insert(0, "_oci_row_id", np.arange(96))
    definitions = [feature(k) for k in ["c", "m", "noise", "other"]]
    splits = [
        {"inner_fold": i, "fit_row_ids": tr.tolist(), "heldout_row_ids": va.tolist()}
        for i, (tr, va) in enumerate(KFold(2, shuffle=True, random_state=9).split(extracted), 1)
    ]
    return dict(
        dataset=dataset,
        extracted_fit=extracted,
        definitions=definitions,
        inner_splits=splits,
        treatment_column="treatment",
        outcome_column="outcome",
        outcome_type="continuous",
        seed=12,
        policy=small_policy(),
    )


def role_fixture():
    definitions = [feature(f"f{i}") for i in range(7)]
    definitions[0].update(configured_explicit_feature=True, roles=["confounder"])
    records = []
    for f in definitions:
        for role in ("treatment", "outcome", "effect"):
            records.append(
                {
                    "feature_id": f["feature_id"],
                    "role": role,
                    "status": "ok",
                    "selected": True,
                    "score": 0.2,
                }
            )
    report = {
        "policy": small_policy().public_dict(),
        "multi_model_evidence": aggregate_evidence(
            [{"family": "univariable", "inner_fold": 1, "records": records}],
            [f["feature_id"] for f in definitions],
        ),
        "row_values": "DO_NOT_LEAK",
        "dataset_path": "DO_NOT_LEAK",
        "oracle": "DO_NOT_LEAK",
    }
    return definitions, report


def test_concept_review_evidence_uses_allowlisted_aggregate_inputs_only():
    definitions, report = role_fixture()
    definitions[1].update(oracle_role="DO_NOT_LEAK", data_generation_notes="DO_NOT_LEAK")
    policy = Stage2RoleAdjudicationConfig()
    evidence = build_multi_model_role_evidence(
        definitions=definitions, statistical_report=report, policy=policy
    )
    rendered = json.dumps(evidence)
    assert "DO_NOT_LEAK" not in rendered
    for key in ("ground_truth", "source_dataset_path", "oracle_role", "data_generation_notes", "row_values"):
        assert key not in rendered
    assert evidence["evidence_boundary"]["outer_heldout_rows_are_excluded"] is True
    assert evidence["evidence_boundary"]["oracle_columns_are_excluded"] is True
    assert [c["feature_id"] for c in evidence["candidates"]] == [f["feature_id"] for f in definitions]
    locked = evidence["candidates"][0]["definition"]
    assert locked["investigator_locked"] is True and locked["configured_roles"] == ["confounder"]
    with pytest.raises(ValueError, match="exactly once"):
        build_multi_model_role_evidence(
            definitions=definitions[:-1], statistical_report=report, policy=policy
        )
