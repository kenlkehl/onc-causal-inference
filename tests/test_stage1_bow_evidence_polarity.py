"""BoW evidence: no "negative" lists from unsigned importances, no duplicates (B8)."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest
import torch

from oci.config import (
    AppliedInferenceConfig,
    BoWViewConfig,
    ForestScientificConfig,
    ModelArchitectureConfig,
    MultiModelForestConfig,
)
from oci.inference.multi_model_forest_stage1 import MultiModelForestStage1Runner
from oci.inference.plain_handoff_stage2_evidence import _extract_sparse_occurrences

pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")


def _dataset(outcome_type: str) -> pd.DataFrame:
    rng = np.random.default_rng(17)
    vocab = [f"word{index}" for index in range(30)]
    rows = []
    for index in range(48):
        treatment = index % 2
        marker = rng.random() < 0.5
        words = list(rng.choice(vocab, size=12))
        if marker:
            words += ["egfr", "mutation", "positive"]
        if treatment:
            words += ["prior", "chemotherapy"]
        rng.shuffle(words)
        signal = 0.2 + 0.3 * marker * treatment + 0.2 * treatment
        outcome = (
            float(signal + rng.normal(scale=0.1))
            if outcome_type == "continuous"
            else int(rng.random() < signal)
        )
        rows.append(
            {
                "clinical_text": " ".join(words),
                "treatment_indicator": treatment,
                "outcome_indicator": outcome,
            }
        )
    return pd.DataFrame(rows)


def _config(outcome_type: str) -> AppliedInferenceConfig:
    views = [
        BoWViewConfig(name="linear_1_2", bow_model="linear", ngram_range_max=2, min_df=1),
        BoWViewConfig(
            name="extratrees_1_2",
            bow_model="extratrees",
            ngram_range_max=2,
            min_df=1,
            forest_scientific=ForestScientificConfig(n_estimators=20),
        ),
    ]
    config = AppliedInferenceConfig(
        outcome_type=outcome_type,
        architecture=ModelArchitectureConfig(
            model_type="multi_model_forest",
            multi_model_forest=MultiModelForestConfig(
                feature_discovery_methods=["bow"],
                matched_pair_uplift_enabled=False,
                bow_views=views,
                nuisance_folds=2,
                effect_folds=2,
                top_n_features=5,
            ),
        ),
    )
    config.seed = 5
    return config


def _handoff_row(tmp_path, outcome_type: str) -> dict:
    dataset = _dataset(outcome_type)
    runner = MultiModelForestStage1Runner(
        dataset=dataset,
        config=_config(outcome_type),
        output_path=tmp_path / "worker" / "predictions.parquet",
        device=torch.device("cpu"),
        num_workers=1,
    )
    return runner.build_discovery_handoff_row(
        train_idx=np.arange(36),
        heldout_idx=np.arange(36, 48),
        fold_key=1,
        outer_fold=1,
        scope="full_outer_train",
    )


@pytest.mark.parametrize("outcome_type", ["binary", "continuous"])
def test_tree_views_publish_no_negative_lists_and_evidence_is_not_duplicated(
    tmp_path, outcome_type
):
    row = _handoff_row(tmp_path, outcome_type)
    # Exact row provenance lets Stage 2 verify the context by row identity.
    assert row["fit_row_ids"] == list(range(36))
    assert row["heldout_row_ids"] == list(range(36, 48))
    importance = row["importance"]
    views = {view["view_name"]: view for view in importance["views"]}

    linear = views["linear_1_2"]
    for target in ("treatment", "outcome", "pseudo_target"):
        assert linear["score_polarity"][target] == "signed"
        assert linear[f"{target}_positive"] and linear[f"{target}_negative"]

    trees = views["extratrees_1_2"]
    for target in ("treatment", "outcome"):
        assert trees["score_polarity"][target] == "unsigned_importance"
        assert f"{target}_negative" not in trees
        assert trees[f"{target}_positive"]
    # Binary effect importances come from the signed DINA model regardless of
    # the view's learner; continuous effects use the unsigned tree regressor.
    effect_signed = outcome_type == "binary"
    assert ("pseudo_target_negative" in trees) is effect_signed
    assert trees["score_polarity"]["pseudo_target"] == (
        "signed" if effect_signed else "unsigned_importance"
    )

    # One cross-view consensus list and a metadata-only ensemble_r record.
    assert "phrase_features" not in importance
    assert importance["phrase_consensus"]
    ensemble = importance["ensemble_r"]
    assert set(ensemble) == {"target_source", "nuisance_sources", "pseudo_target_construction"}
    if outcome_type == "binary":
        assert ensemble["target_source"] == "per_view_cross_fitted_dina_nuisance"
        assert "DINA" in ensemble["pseudo_target_construction"]
    else:
        assert ensemble["target_source"] == "ensemble_mean_nuisance"
        assert "(Y - mean_m_hat) / (T - mean_e_hat)" in ensemble["pseudo_target_construction"]

    occurrences = _extract_sparse_occurrences(
        {"source": "text_models", "outer_fold": 1, "inner_fold": None, "scope": "full_outer_train"},
        row,
        handoff_row=1,
    )
    paths = [occurrence["reference"]["json_path"] for occurrence in occurrences]
    assert not any(path.startswith("importance.ensemble_r") for path in paths)
    assert not any(
        path.startswith(("importance.views[1].treatment_negative", "importance.views[1].outcome_negative"))
        for path in paths
    )
    consensus_counts = Counter(
        (occurrence["text"], occurrence["architecture"])
        for occurrence, path in zip(occurrences, paths)
        if ".phrase_" in path
    )
    assert consensus_counts
    assert set(consensus_counts.values()) == {1}
