"""The ``matched_pair_uplift`` lane: matching, pair models, evidence, metrics."""

from __future__ import annotations


import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit

from oci.config import (
    AppliedInferenceConfig,
    BoWViewConfig,
    ModelArchitectureConfig,
    MultiModelForestConfig,
)
from oci.inference.multi_model_pair_uplift import (
    MatchedPairSettings,
    build_matched_pairs,
    distinct_vectorizer_groups,
    run_matched_pair_lane,
)

VOCABULARY = [f"tok{index:02d}" for index in range(60)]
UNIGRAM = BoWViewConfig(name="unigram", ngram_range_min=1, ngram_range_max=1, min_df=2)


def _settings(outcome_type: str = "binary", **overrides) -> MatchedPairSettings:
    config = MultiModelForestConfig(bow_views=[UNIGRAM], **overrides)
    return MatchedPairSettings.from_config(config, outcome_type=outcome_type, top_n=10)


def _simulate(seed: int, *, modifier: float, outcome_type: str = "binary", n: int = 1200):
    """Texts with a confounder ("smoker") and a candidate modifier ("kras")."""

    rng = np.random.default_rng(seed)
    present = rng.random((n, len(VOCABULARY))) < 0.3
    smoker = rng.random(n) < 0.4
    kras = rng.random(n) < 0.5
    texts = []
    for index in range(n):
        words = [word for word, keep in zip(VOCABULARY, present[index]) if keep]
        if smoker[index]:
            words.append("smoker")
        if kras[index]:
            words.append("kras")
        rng.shuffle(words)
        texts.append(" ".join(words))
    propensity = expit(-0.2 + 0.8 * smoker)
    treatment = (rng.random(n) < propensity).astype(float)
    if outcome_type == "binary":
        prognostic = expit(-0.3 + 0.8 * smoker)
        effect = 0.3 + modifier * kras
        outcome = (rng.random(n) < expit(logit(prognostic) + treatment * effect)).astype(float)
    else:
        prognostic = 0.5 * smoker
        outcome = prognostic + treatment * (1.0 + modifier * kras) + rng.normal(size=n)
    frame = pd.DataFrame({"_oci_row_id": np.arange(n)})
    return frame, texts, outcome, treatment, propensity, prognostic


def _run(seed: int, *, modifier: float, outcome_type: str = "binary", n: int = 1200):
    frame, texts, outcome, treatment, propensity, prognostic = _simulate(
        seed, modifier=modifier, outcome_type=outcome_type, n=n
    )
    test = pd.DataFrame({"_oci_row_id": [n, n + 1]})
    return run_matched_pair_lane(
        train_df=frame,
        test_df=test,
        texts_train=texts,
        texts_test=["kras smoker tok01", "tok01 tok02"],
        y_train=outcome,
        t_train=treatment,
        propensity=propensity,
        prognostic=prognostic,
        score_sources={"propensity": "simulated", "prognostic": "simulated"},
        vectorizer_groups=distinct_vectorizer_groups([UNIGRAM]),
        settings=_settings(outcome_type),
        outer_fold=1,
        seed=seed,
    )


def _ranked_terms(view):
    rows = [*view["effect_modification_positive"], *view["effect_modification_negative"]]
    return sorted(rows, key=lambda row: row["rank"])


def test_matching_stays_within_both_calipers_without_fallback():
    treatment = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=float)
    propensity = np.array([0.30, 0.31, 0.70, 0.50, 0.30, 0.33, 0.90, 0.52])
    prognostic = np.array([0.20, 0.21, 0.40, 0.50, 0.24, 0.21, 0.40, 0.90])
    pairs, stats = build_matched_pairs(
        row_ids=np.arange(10, 18),
        treatment=treatment,
        propensity=propensity,
        prognostic=prognostic,
        propensity_caliper=0.05,
        prognostic_caliper=0.05,
    )

    # Rows 2 and 3 have no control within both calipers and stay unmatched.
    assert set(pairs["treated_pos"]) == {0, 1}
    assert stats["n_pairs"] == 2 and stats["n_unmatched_treated"] == 2
    assert stats["nearest_neighbour_fallback"] is False
    assert (pairs["abs_delta_propensity"] <= 0.05).all()
    assert (pairs["abs_delta_prognostic"] <= 0.05).all()
    # 1:1 without replacement, and each treated row takes its closest match
    # (row 1 is closer to control 5 than to control 4).
    assert pairs["control_pos"].is_unique
    assert dict(zip(pairs["treated_pos"], pairs["control_pos"])) == {0: 4, 1: 5}
    assert pairs["treated_row_id"].tolist() == [10, 11]


def test_binary_modifier_is_recovered_with_the_correct_sign():
    result = _run(0, modifier=1.5)
    view = result.importance["views"][0]

    assert result.importance["status"] == "fitted"
    terms = _ranked_terms(view)
    assert terms[0]["feature"] == "kras" and terms[0]["score"] > 0
    assert view["effect_scale"] == "conditional_log_odds_ratio"
    # Predictions use each test patient's own features.
    tau = result.prediction_frames[0]["tau_hat_matched_pair"].to_numpy()
    assert tau[0] > tau[1]
    assert set(result.prediction_frames[0]["split_role"]) == {"test_outer_train_fit"}


def test_continuous_prognostic_score_is_out_of_fold():
    from oci.inference.multi_model_pair_uplift import crossfit_prognostic_score
    from oci.inference.sparse_text_modeling import bow_vectorizer_params

    frame, texts, outcome, treatment, _propensity, _prognostic = _simulate(
        2, modifier=1.0, outcome_type="continuous", n=240
    )
    kwargs = dict(
        vectorizer_params=bow_vectorizer_params(UNIGRAM),
        folds=4,
        l1_ratio=0.5,
        cv_folds=3,
        max_iter=5000,
        seed=9,
    )
    first = crossfit_prognostic_score(texts, outcome, treatment, **kwargs)
    control = int(np.flatnonzero(treatment == 0)[0])
    changed = outcome.copy()
    changed[control] += 50.0
    second = crossfit_prognostic_score(texts, changed, treatment, **kwargs)

    assert np.isfinite(first["prognostic"]).all()
    # A row's own outcome never reaches its own prediction.
    assert second["prognostic"][control] == pytest.approx(first["prognostic"][control])
    assert not np.allclose(second["prognostic"], first["prognostic"])
    assert first["audit"]["folds"] == 4


@pytest.mark.parametrize("outcome_type", ["binary"])
def test_stage1_context_runs_the_lane_from_out_of_fold_scores(tmp_path, outcome_type):
    from oci.config import EmbeddingContrastDiscoveryConfig
    from oci.inference.multi_model_forest_stage1 import run_multi_model_forest_handoff_context
    from oci.inference.multi_model_pair_uplift import (
        BINARY_PROGNOSTIC_SOURCE,
        CONTINUOUS_PROGNOSTIC_SOURCE,
        PROPENSITY_SOURCE,
    )

    frame, texts, outcome, treatment, _propensity, _prognostic = _simulate(
        5, modifier=1.5, outcome_type=outcome_type, n=360
    )
    dataset = pd.DataFrame(
        {
            "clinical_text": texts,
            "treatment_indicator": treatment.astype(int),
            "outcome_indicator": outcome if outcome_type == "continuous" else outcome.astype(int),
            "true_ite": np.zeros(len(texts)),
        }
    )
    config = AppliedInferenceConfig(
        seed=3,
        outcome_type=outcome_type,
        architecture=ModelArchitectureConfig(
            model_type="multi_model_forest",
            multi_model_forest=MultiModelForestConfig(
                nuisance_folds=3,
                effect_folds=3,
                top_n_features=10,
                htr_evidence_enabled=False,
                bow_views=[UNIGRAM],
                embedding_contrast=EmbeddingContrastDiscoveryConfig(
                    enabled=False, disable_reason="matched-pair lane test"
                ),
            ),
        ),
    )
    rows = np.arange(len(dataset))
    row = run_multi_model_forest_handoff_context(
        dataset=dataset,
        config=config,
        context={
            "train_idx": rows[:300],
            "heldout_idx": rows[300:],
            "fold_key": 1,
            "outer_fold": 1,
            "scope": "full_outer_train",
        },
        handoff_dir=tmp_path,
        device="cpu",
        num_workers=1,
    )
    lane = row["importance"]["matched_pair_uplift"]
    assert lane["outcome_type"] == outcome_type
    assert lane["status"] == "fitted"
    assert lane["score_sources"]["propensity"]["source"] == PROPENSITY_SOURCE
    assert lane["score_sources"]["prognostic"]["source"] == (
        BINARY_PROGNOSTIC_SOURCE if outcome_type == "binary" else CONTINUOUS_PROGNOSTIC_SOURCE
    )
    # Matching and fitting use only the context's training rows.
    assert lane["matching"]["n_rows"] == 300
    predictions = pd.read_parquet(
        tmp_path / "worker_artifacts" / "fold_000001" / "predictions.parquet"
    )
    pair_rows = predictions.loc[predictions["architecture"] == "matched_pair_uplift"]
    assert set(pair_rows["split_role"]) == {"test_outer_train_fit"}
    assert sorted(pair_rows["_oci_row_id"]) == list(range(300, 360))
    assert pair_rows["tau_hat_matched_pair"].notna().all()
    assert row["metrics"]["matched_pair_status"] == "fitted"
