import json

import numpy as np
import pandas as pd
import pytest

from oci.config import (
    AppliedInferenceConfig,
    BoWViewConfig,
    ModelArchitectureConfig,
    MultiModelForestConfig,
    TfidfTopicDiscoveryConfig,
    TfidfVectorizerScientificConfig,
)
from oci.inference import tfidf_topic_stage1
from oci.inference.tfidf_topic_stage1 import (
    _build_tfidf_worker_context_spec,
    run_tfidf_topic_stage1,
)

pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")


def _dataset(rows_count: int = 96) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    vocab = [f"term{index}" for index in range(60)]
    rows = []
    for _ in range(rows_count):
        treatment = int(rng.random() < 0.5)
        marker = rng.random() < 0.5
        outcome = int(rng.random() < (0.3 + 0.3 * marker * treatment + 0.1 * treatment))
        words = list(rng.choice(vocab, size=40))
        if marker:
            words += ["egfr", "mutation", "positive"]
        if treatment:
            words += ["osimertinib", "started"]
        if outcome:
            words += ["progression", "noted"]
        rng.shuffle(words)
        rows.append(
            {
                "clinical_text": " ".join(words),
                "treatment_indicator": treatment,
                "outcome_indicator": outcome,
            }
        )
    return pd.DataFrame(rows)


def _config(*, orphan: bool) -> AppliedInferenceConfig:
    views = [
        BoWViewConfig(
            name=name,
            bow_model="linear",
            ngram_range_min=1,
            ngram_range_max=ngram_max,
            min_df=1,
            max_df=1.0,
            max_features=2000,
        )
        for name, ngram_max in (("linear_1_3", 3), ("linear_1_2", 2))
    ]
    topic = TfidfTopicDiscoveryConfig(
        vectorizer_scientific=TfidfVectorizerScientificConfig(
            min_df=1,
            max_df=1.0,
            max_features=2000,
        ),
        topic_count=3,
        terms_per_topic=5,
        stability_repeats=2,
        orphan_ngram_enabled=orphan,
        orphan_ngram_min_abs_fit_score=0.5,
        orphan_ngram_min_selected_clusters=2,
        orphan_ngram_max_selected_clusters=4,
        orphan_ngram_bootstrap_repeats=20,
    )
    config = AppliedInferenceConfig(
        dataset_path="in_memory",
        outcome_type="binary",
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        cv_folds=2,
        architecture=ModelArchitectureConfig(
            model_type="multi_model_forest",
            multi_model_forest=MultiModelForestConfig(
                candidate_consistency_inner_folds=2,
                nuisance_folds=2,
                bow_views=views,
                tfidf_topic=topic,
                cpus_total=1,
                outer_parallel_backend="threads",
            ),
        ),
    )
    config.seed = 11
    return config


def _run(tmp_path, config, *, rows_count: int = 96):
    run_tfidf_topic_stage1(
        dataset=_dataset(rows_count),
        config=config,
        output_path=tmp_path / "predictions.parquet",
        artifact_dir=tmp_path,
        handoff_path=tmp_path / "evidence.jsonl",
    )
    with (tmp_path / "evidence.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def test_topic_contexts_are_complete_and_resume_without_refitting(tmp_path, monkeypatch):
    config = _config(orphan=False)
    rows = _run(tmp_path, config)

    assert [(row["outer_fold"], row["scope"]) for row in rows] == [
        (1, "candidate_selection_inner_fit"),
        (1, "candidate_selection_inner_fit"),
        (1, "full_outer_train"),
        (2, "candidate_selection_inner_fit"),
        (2, "candidate_selection_inner_fit"),
        (2, "full_outer_train"),
    ]
    for row in rows:
        discovery = row["discovery"]
        assert discovery["fit_row_ids"] == row["fit_row_ids"]
        assert "effect_orphan_ngram_branch" not in discovery
        assert discovery["topic_banks"]["treatment"]["topics"]
    predictions = pd.read_parquet(tmp_path / "predictions.parquet")
    assert set(predictions["prediction_scope"]) == {"external_heldout"}
    assert sorted(predictions["_oci_row_id"]) == list(range(len(_dataset())))

    def refit(**_kwargs):
        raise AssertionError("complete contexts must be reused")

    monkeypatch.setattr(tfidf_topic_stage1, "fit_tfidf_topic_context", refit)
    assert _run(tmp_path, config) == rows


def test_orphan_lane_selects_clusters_from_fit_rows_only(tmp_path):
    # The orphan lane nests three levels of cross-fitting; 96 rows leave some
    # DINA nuisance subfolds without a treatment arm for some run seeds, which
    # correctly fails closed. 144 rows keep every subfold feasible.
    rows = _run(tmp_path, _config(orphan=True), rows_count=144)

    for row in rows:
        discovery = row["discovery"]
        branch = discovery["effect_orphan_ngram_branch"]
        assert branch["status"] == "completed"
        assert branch["selected_cluster_ids"]
        assert discovery["orphan_ngram_label_policy"] == "nested_fit_calibration"
        assert discovery["registered_heldout_labels_accessed"] is False
        nesting = discovery["selection_nesting"]
        assert set(nesting["model_fit_row_ids"]) | set(nesting["calibration_row_ids"]) == set(
            row["fit_row_ids"]
        )
        assert not set(nesting["calibration_row_ids"]) & set(row["heldout_row_ids"])


def test_worker_specs_never_carry_registered_heldout_labels():
    data = _dataset()
    data["_oci_row_id"] = np.arange(len(data))
    spec = _build_tfidf_worker_context_spec(
        outer_fold=1,
        inner_fold=None,
        scope="full_outer_train",
        fold_key=1,
        fit_df=data.iloc[:60],
        heldout_df=data.iloc[60:],
        scope_id="outer_001_full_train",
        config=_config(orphan=True),
    )
    assert list(spec["heldout_df"].columns) == ["_oci_row_id", "clinical_text"]
