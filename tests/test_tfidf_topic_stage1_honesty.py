import numpy as np
import pandas as pd

from oci.config import (
    AppliedInferenceConfig,
    ModelArchitectureConfig,
    MultiModelForestConfig,
)
from oci.inference.tfidf_topic_stage1 import (
    _outer_split_plan,
    make_joint_treatment_outcome_splits,
    tfidf_topic_dataset_fingerprints,
    tfidf_topic_stage1_config_hash,
)


def _dataset() -> pd.DataFrame:
    rows = []
    for treatment in (0, 1):
        for outcome in (0, 1):
            for replicate in range(9):
                rows.append(
                    {
                        "clinical_text": (
                            f"baseline record arm {treatment} response {outcome} "
                            f"patient {replicate}"
                        ),
                        "treatment_indicator": treatment,
                        "outcome_indicator": outcome,
                        "true_ite_prob": float(replicate) / 100.0,
                    }
                )
    return pd.DataFrame(rows)


def _config(seed: int = 17) -> AppliedInferenceConfig:
    config = AppliedInferenceConfig(
        dataset_path="in_memory",
        outcome_type="binary",
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        cv_folds=3,
        architecture=ModelArchitectureConfig(
            model_type="multi_model_forest",
            multi_model_forest=MultiModelForestConfig(
                candidate_consistency_inner_folds=3,
                nuisance_folds=3,
            ),
        ),
    )
    config.seed = int(seed)
    return config


def test_joint_treatment_outcome_split_is_balanced_and_seeded():
    data = _dataset()
    splits, metadata = make_joint_treatment_outcome_splits(
        data,
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        outcome_type="binary",
        n_splits=3,
        seed=17,
    )
    repeated, repeated_metadata = make_joint_treatment_outcome_splits(
        data,
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        outcome_type="binary",
        n_splits=3,
        seed=17,
    )
    changed_seed, _ = make_joint_treatment_outcome_splits(
        data,
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        outcome_type="binary",
        n_splits=3,
        seed=18,
    )

    assert metadata == repeated_metadata
    assert metadata["method"] == "stratified_joint_treatment_outcome"
    assert metadata["fallback_reason"] is None
    assert all(np.array_equal(left[1], right[1]) for left, right in zip(splits, repeated))
    assert any(not np.array_equal(left[1], right[1]) for left, right in zip(splits, changed_seed))
    for _, heldout in splits:
        cell_counts = (
            data.iloc[heldout].groupby(["treatment_indicator", "outcome_indicator"]).size()
        )
        assert cell_counts.to_dict() == {(0, 0): 3, (0, 1): 3, (1, 0): 3, (1, 1): 3}


def test_joint_split_falls_back_only_when_a_joint_cell_is_too_small():
    data = _dataset().iloc[:10].copy()
    data.loc[data.index[-1], ["treatment_indicator", "outcome_indicator"]] = [1, 1]
    splits, metadata = make_joint_treatment_outcome_splits(
        data,
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        outcome_type="binary",
        n_splits=3,
        seed=23,
    )

    assert len(splits) == 3
    assert metadata["method"] == "kfold_fallback"
    assert metadata["minimum_joint_stratum_count"] < 3
    assert metadata["fallback_reason"] == ("minimum_joint_stratum_count_below_requested_folds")


def test_outer_plan_uses_config_seed_instead_of_a_fixed_constant():
    data = _dataset()
    first, first_metadata = _outer_split_plan(data, _config(seed=101))
    second, second_metadata = _outer_split_plan(data, _config(seed=102))

    assert first_metadata["seed"] == 101
    assert second_metadata["seed"] == 102
    assert any(not np.array_equal(left[1], right[1]) for left, right in zip(first, second))


def test_stage1_hash_tracks_model_content_row_order_seed_and_split_semantics():
    data = _dataset()
    config = _config(seed=31)
    identity = tfidf_topic_dataset_fingerprints(data, config)
    baseline_hash = tfidf_topic_stage1_config_hash(config, data)

    oracle_only_change = data.copy()
    oracle_only_change["true_ite_prob"] += 100.0
    assert tfidf_topic_stage1_config_hash(config, oracle_only_change) == baseline_hash

    content_change = data.copy()
    content_change.loc[0, "clinical_text"] += " changed"
    assert tfidf_topic_stage1_config_hash(config, content_change) != baseline_hash

    reordered = data.iloc[::-1].reset_index(drop=True)
    reordered_identity = tfidf_topic_dataset_fingerprints(reordered, config)
    assert reordered_identity["content_fingerprint"] == identity["content_fingerprint"]
    assert reordered_identity["ordered_row_fingerprint"] != identity["ordered_row_fingerprint"]
    assert tfidf_topic_stage1_config_hash(config, reordered) != baseline_hash

    changed_seed = _config(seed=32)
    assert tfidf_topic_stage1_config_hash(changed_seed, data) != baseline_hash

    changed_inner_folds = _config(seed=31)
    changed_inner_folds.architecture.multi_model_forest.candidate_consistency_inner_folds = 2
    assert tfidf_topic_stage1_config_hash(changed_inner_folds, data) != baseline_hash
