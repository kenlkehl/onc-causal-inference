"""The ``matched_pair_uplift`` lane: matching, pair models, evidence, metrics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.special import expit, logit

from oci.config import (
    AppliedInferenceConfig,
    BoWViewConfig,
    ModelArchitectureConfig,
    MultiModelForestConfig,
)
from oci.evaluation.stage1 import _matched_pair_lane_metrics, evaluate_stage1_architectures
from oci.inference.multi_model_pair_uplift import (
    MatchedPairSettings,
    build_matched_pairs,
    constant_intercept,
    distinct_vectorizer_groups,
    fit_penalized_pair_model,
    hopcroft_karp,
    pair_analysis_data,
    resolve_prognostic_caliper,
    run_matched_pair_lane,
    summarize_pairs,
)
from oci.inference.plain_handoff_stage2_evidence import extract_stage1_architecture_occurrences
from oci.inference.stage1_architecture_artifacts import materialize_stage1_architecture_artifacts
from oci.inference.stage1_architectures import (
    DEFAULT_STAGE1_ARCHITECTURES,
    MATCHED_PAIR_EVIDENCE_SCHEMA_VERSION,
    MATCHED_PAIR_UPLIFT,
    STAGE1_ARCHITECTURES,
    legacy_enabled_stage1_architectures,
    unavailable_explicit_architectures,
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


def test_matching_uses_only_the_rows_it_is_given():
    pairs, stats = build_matched_pairs(
        row_ids=[0, 1],
        treatment=np.array([1.0, 1.0]),
        propensity=np.array([0.4, 0.5]),
        prognostic=np.array([0.4, 0.5]),
        propensity_caliper=0.05,
        prognostic_caliper=0.05,
    )
    assert pairs.empty and stats["n_pairs"] == 0


def test_hopcroft_karp_finds_a_maximum_matching():
    rng = np.random.default_rng(3)
    for _ in range(25):
        n_left, n_right = rng.integers(5, 40, size=2)
        dense = rng.random((n_left, n_right)) < 0.12
        adjacency = [np.flatnonzero(row).tolist() for row in dense]
        pair_left, pair_right = hopcroft_karp(adjacency, int(n_right))
        matched = [(left, right) for left, right in enumerate(pair_left) if right >= 0]
        assert all(dense[left, right] for left, right in matched)
        assert len({right for _left, right in matched}) == len(matched)
        assert all(pair_right[right] == left for left, right in matched)
        reference = maximum_bipartite_matching(sparse.csr_matrix(dense), perm_type="column")
        assert len(matched) == int(np.sum(reference >= 0))


def test_prognostic_caliper_scale_depends_on_outcome_type():
    prognostic = np.array([0.0, 1.0, 2.0, 3.0])
    binary = resolve_prognostic_caliper(_settings("binary"), prognostic / 10)
    assert binary["absolute"] == pytest.approx(0.05) and binary["scale"] == "probability"
    continuous = resolve_prognostic_caliper(_settings("continuous"), prognostic)
    assert continuous["absolute"] == pytest.approx(0.2 * np.std(prognostic))
    with pytest.raises(ValueError, match="probability scale"):
        AppliedInferenceConfig(
            outcome_type="binary",
            architecture=ModelArchitectureConfig(
                multi_model_forest=MultiModelForestConfig(matched_pair_prognostic_caliper=1.5)
            ),
        )


def test_discordant_pairs_and_d_coding_and_offset():
    pairs = pd.DataFrame(
        {
            "treated_pos": [0, 1, 2, 3],
            "control_pos": [4, 5, 6, 7],
            "treated_row_id": [0, 1, 2, 3],
            "control_row_id": [4, 5, 6, 7],
            "prognostic_treated": [0.2, 0.3, 0.4, 0.5],
            "prognostic_control": [0.25, 0.35, 0.45, 0.55],
            "abs_delta_propensity": [0.0] * 4,
            "abs_delta_prognostic": [0.05] * 4,
        }
    )
    # Pairs: (1,0) treated event only, (0,1) control event only, (1,1), (0,0).
    outcome = np.array([1, 0, 1, 0, 0, 1, 1, 0], dtype=float)
    analysis = pair_analysis_data(pairs, outcome, binary=True)

    assert analysis.pairs["treated_pos"].tolist() == [0, 1]
    assert analysis.target.tolist() == [1.0, 0.0]
    np.testing.assert_allclose(
        analysis.offset,
        logit(np.array([0.2, 0.3])) - logit(np.array([0.25, 0.35])),
    )
    summary = summarize_pairs(pairs, outcome, binary=True)
    assert summary["n_discordant_pairs"] == 2
    assert summary["n10_treated_event_only"] == 1
    assert summary["n01_control_event_only"] == 1
    assert summary["mcnemar_log_odds_ratio"] == pytest.approx(0.0)

    continuous = pair_analysis_data(pairs, outcome * 3.0, binary=False)
    assert len(continuous.pairs) == 4
    np.testing.assert_allclose(continuous.target, [3.05, -2.95, 0.05, 0.05])


def test_offset_enters_the_conditional_likelihood():
    rng = np.random.default_rng(0)
    offset = rng.normal(scale=2.0, size=400)
    # D depends only on the offset plus a common log odds ratio of 0.4.
    d = (rng.random(400) < expit(offset + 0.4)).astype(float)
    assert constant_intercept(d, offset, "logistic") == pytest.approx(0.4, abs=0.2)
    assert constant_intercept(d, np.zeros(400), "logistic") != pytest.approx(
        constant_intercept(d, offset, "logistic"), abs=1e-3
    )
    features = sparse.csr_matrix(rng.normal(size=(400, 5)))
    fitted = fit_penalized_pair_model(
        features,
        np.asarray(features.mean(axis=0)).ravel(),
        d,
        offset,
        loss="logistic",
        l1_ratio=0.5,
        cv_folds=5,
        max_iter=5000,
        seed=1,
    )
    assert fitted["intercept"] == pytest.approx(0.4, abs=0.25)
    assert np.count_nonzero(fitted["beta"]) <= 1


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


def test_no_modification_yields_the_intercept_only_most_of_the_time():
    intercept_only = [
        _run(seed, modifier=0.0).importance["views"][0]["n_nonzero_terms"] == 0
        for seed in range(5)
    ]
    assert sum(intercept_only) >= 4


def test_continuous_modifier_is_recovered():
    result = _run(1, modifier=2.0, outcome_type="continuous")
    view = result.importance["views"][0]

    assert result.importance["method"] == "elastic_net_regression_of_adjusted_paired_differences"
    assert _ranked_terms(view)[0]["feature"] == "kras"
    assert _ranked_terms(view)[0]["score"] > 0
    # True average effect is 1 + 2 * P(kras) = 2.
    assert view["average_effect"] == pytest.approx(2.0, abs=0.4)
    assert view["validation"]["heldout_mse_reduction"] > 0


def test_below_minimum_pairs_emits_no_evidence_and_records_the_reason():
    result = _run(0, modifier=1.5, n=40)
    importance = result.importance

    assert importance["status"] == "insufficient_discordant_pairs"
    assert "matched_pair_min_discordant_pairs" in importance["reason"]
    assert importance["views"] == [] and result.prediction_frames == []
    row = {
        "source": "text_models",
        "outer_fold": 1,
        "scope": "full_outer_train",
        "evidence": {"importance": {"matched_pair_uplift": importance}},
    }
    occurrences = extract_stage1_architecture_occurrences(
        [row, _other_lane_row()], included_architectures=[MATCHED_PAIR_UPLIFT]
    )
    assert occurrences[1] == ()

    continuous = _run(0, modifier=1.0, outcome_type="continuous", n=40)
    assert continuous.importance["status"] == "insufficient_pairs"
    assert "matched_pair_min_pairs" in continuous.importance["reason"]


def _other_lane_row():
    return {
        "source": "text_models",
        "outer_fold": 1,
        "scope": "full_outer_train",
        "evidence": {
            "importance": {
                "views": [
                    {"view_name": "v", "treatment_positive": [{"feature": "ecog", "score": 1}]}
                ]
            }
        },
    }


def test_evidence_has_no_control_side_terms():
    result = _run(0, modifier=1.5)
    rendered = json.dumps(result.importance)
    assert "pair_side" not in rendered
    assert "control::" not in rendered and "treated::" not in rendered
    assert result.importance["schema_version"] == MATCHED_PAIR_EVIDENCE_SCHEMA_VERSION

    row = {
        "source": "text_models",
        "outer_fold": 1,
        "scope": "full_outer_train",
        "evidence": {"importance": {"matched_pair_uplift": result.importance}},
    }
    occurrences = extract_stage1_architecture_occurrences([row])[1]
    assert occurrences
    assert {item["architecture"] for item in occurrences} == {MATCHED_PAIR_UPLIFT}
    assert all("pair_side" not in item["details"] for item in occurrences)
    kras = [item for item in occurrences if item["text"] == "kras"]
    assert kras and kras[0]["polarity"] == "positive"
    assert kras[0]["axes"] == ["matched_pair"]


def test_stage2_rejects_earlier_treated_control_payloads():
    old = {
        "source": "text_models",
        "outer_fold": 1,
        "scope": "full_outer_train",
        "evidence": {
            "importance": {
                "matched_pair_uplift": {
                    "views": [
                        {
                            "view_name": "pair",
                            "uplift_delta_logit_positive": [
                                {"feature": "treated::response", "score": 1.0}
                            ],
                        }
                    ]
                }
            }
        },
    }
    with pytest.raises(ValueError, match="rebuild Stage 1"):
        extract_stage1_architecture_occurrences([old])
    htr = {
        "source": "text_models",
        "outer_fold": 1,
        "scope": "full_outer_train",
        "evidence": {
            "importance": {},
            "htr_evidence": {"pair_uplift": {"attention": [{"chunk_text": "pair witness"}]}},
        },
    }
    with pytest.raises(ValueError, match="rebuild Stage 1"):
        extract_stage1_architecture_occurrences([htr])


def test_registry_runs_eight_lanes_and_accepts_all_for_continuous_outcomes():
    applied = AppliedInferenceConfig(outcome_type="continuous")
    selected = legacy_enabled_stage1_architectures(applied, outcome_type="continuous")
    assert selected == DEFAULT_STAGE1_ARCHITECTURES
    assert len(selected) == 8 and MATCHED_PAIR_UPLIFT in selected
    assert unavailable_explicit_architectures(
        STAGE1_ARCHITECTURES, applied, outcome_type="continuous"
    ) == ()
    disabled = AppliedInferenceConfig(
        outcome_type="continuous",
        architecture=ModelArchitectureConfig(
            multi_model_forest=MultiModelForestConfig(bow_discovery_enabled=False)
        ),
    )
    assert MATCHED_PAIR_UPLIFT in unavailable_explicit_architectures(
        (MATCHED_PAIR_UPLIFT,), disabled, outcome_type="continuous"
    )


def test_one_fit_per_distinct_vectorizer():
    views = MultiModelForestConfig().bow_views
    groups = distinct_vectorizer_groups(views)
    assert [group["view_names"] for group in groups] == [
        ["linear_unigram_c0p5"],
        ["linear_1_2", "random_forest_1_2"],
        ["linear_1_3", "extratrees_1_3"],
        ["linear_2_4_min_df3"],
    ]


def test_native_metrics_are_read_from_the_frozen_payload(tmp_path: Path):
    result = _run(0, modifier=1.5)
    payload = {
        "schema_version": "multi_model_agentic_discovery_handoff_v1",
        "outer_fold": 1,
        "scope": "full_outer_train",
        "importance": {"matched_pair_uplift": result.importance},
    }
    metrics = {row["metric"]: row for row in _matched_pair_lane_metrics([payload])}
    matching = result.importance["matching"]
    assert metrics["n_matched_pairs"]["value"] == matching["n_pairs"]
    assert metrics["n_discordant_pairs"]["value"] == matching["n_discordant_pairs"]
    assert metrics["n10_treated_event_only"]["value"] == matching["n10_treated_event_only"]
    assert metrics["n01_control_event_only"]["value"] == matching["n01_control_event_only"]
    assert metrics["mcnemar_log_odds_ratio"]["value"] == pytest.approx(
        np.log(matching["n10_treated_event_only"] / matching["n01_control_event_only"])
    )
    for key in ("mean_abs_delta_propensity", "max_abs_delta_prognostic"):
        assert metrics[key]["applicability"] == "applicable"
    assert metrics["max_abs_delta_propensity"]["value"] <= 0.05
    assert metrics["heldout_log_likelihood_gain_per_pair"]["applicability"] == "applicable"
    assert metrics["n_nonzero_terms"]["value"] == result.importance["views"][0]["n_nonzero_terms"]
    assert metrics["n_matched_pairs"]["outer_fold"] == 1

    # End to end: the evaluator hashes the text-model source before oracle data.
    run_dir = tmp_path / "run"
    dataset_path = tmp_path / "dataset.parquet"
    pd.DataFrame(
        {
            "clinical_text": ["a", "b"],
            "treatment_indicator": [0, 1],
            "outcome_indicator": [0, 1],
            "true_ite_prob": [0.1, 0.2],
        }
    ).to_parquet(dataset_path)
    run_dir.mkdir()
    (run_dir / "run_config.json").write_text(
        json.dumps({"dataset": str(dataset_path), "outcome_type": "binary"}), encoding="utf-8"
    )
    source = run_dir / "components" / "text_models" / "evidence.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    raw = {"source": "text_models", "outer_fold": 1, "scope": "full_outer_train", "evidence": payload}
    materialize_stage1_architecture_artifacts(
        output_dir=run_dir,
        raw_handoff_rows=[raw],
        selected_architectures=(MATCHED_PAIR_UPLIFT,),
        source_artifacts={"text_models": source},
        selection_mode="explicit",
    )
    summary = evaluate_stage1_architectures(run_dir=run_dir)
    output = Path(summary["output_dir"])
    manifest = json.loads((output / "evaluation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["consumed_source_sha256"] == {
        "components/text_models/evidence.jsonl": manifest["consumed_source_sha256"][
            "components/text_models/evidence.jsonl"
        ]
    }
    rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert any(
        row["metric"] == "n_matched_pairs" and row["value"] == matching["n_pairs"]
        for row in rows
    )


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


@pytest.mark.parametrize("outcome_type", ["binary", "continuous"])
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
