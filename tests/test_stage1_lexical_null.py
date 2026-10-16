"""Random-chunk null for the Stage 1 lexical recovery metric (S1-29)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oci.evaluation.stage1 import evaluate_stage1_architectures
from oci.evaluation.stage1_lexical_null import (
    TrainingTextCorpus,
    _draw_windows,
    best_token_recall,
    null_seed,
    random_chunk_null,
    summarize_null,
    tokenize,
    window_token_presence,
)
from oci.inference.stage1_architecture_artifacts import (
    materialize_stage1_architecture_artifacts,
)

WORDS = ["ecog", "ecogs", "stage", "staged", "iv", "egfr", "pain", "x9", "ps", "ecog2"]


def _texts(count: int = 30, seed: int = 0) -> list[str]:
    rng = np.random.default_rng(seed)
    output = []
    for _ in range(count):
        words = rng.choice(WORDS, size=int(rng.integers(5, 40)))
        separators = rng.choice([" ", ", ", "-", "\n", " (", ") "], size=len(words))
        output.append("".join(f"{word.upper() if rng.random() < 0.2 else word}{sep}"
                              for word, sep in zip(words, separators)))
    return output


def test_vectorized_window_tokens_match_substring_tokenization():
    texts = _texts()
    corpus = TrainingTextCorpus(texts)
    starts, ends, _unmatched = _draw_windows(
        corpus, range(len(texts)), [3, 5, 9, 17, 40, 400], draws=50, seed=4
    )
    for token in ("ecog", "stage", "iv", "ps", "x9", "egfr"):
        presence = window_token_presence(corpus, starts, ends, token)
        expected = np.vectorize(
            lambda start, end: token in tokenize(corpus.text[start:end])
        )(starts, ends)
        assert np.array_equal(presence, expected), token


def test_draws_are_deterministic_length_matched_and_from_training_rows_only():
    texts = _texts(seed=1)
    corpus = TrainingTextCorpus(texts)
    training = list(range(0, 20))
    lengths = [4, 12, 25]
    first = _draw_windows(corpus, training, lengths, draws=40, seed=9)
    again = _draw_windows(corpus, training, lengths, draws=40, seed=9)
    other = _draw_windows(corpus, training, lengths, draws=40, seed=10)

    assert all(np.array_equal(left, right) for left, right in zip(first[:2], again[:2]))
    assert not np.array_equal(first[0], other[0])
    starts, ends, unmatched = first
    assert unmatched == 0
    assert np.all(ends - starts == np.asarray(lengths)[None, :])
    rows = np.searchsorted(corpus.base, starts, side="right") - 1
    assert set(rows.ravel().tolist()) <= set(training)
    # Windows never cross into the next row.
    assert np.all(ends <= corpus.base[rows] + corpus.lengths[rows])


def test_items_longer_than_every_training_row_use_whole_rows_and_are_counted():
    corpus = TrainingTextCorpus(["short note", "ecog two", "another one"])
    starts, ends, unmatched = _draw_windows(corpus, [0, 1, 2], [500, 3], draws=5, seed=1)
    assert unmatched == 1
    rows = np.searchsorted(corpus.base, starts[:, 0], side="right") - 1
    assert np.all(ends[:, 0] - starts[:, 0] == corpus.lengths[rows])
    assert np.all(ends[:, 1] - starts[:, 1] == 3)


def test_seed_depends_on_run_seed_architecture_and_context():
    kwargs = dict(run_seed=7, architecture="tfidf_topics", outer_fold=1, inner_fold=None,
                  schema_version="v")
    base = null_seed(**kwargs)
    assert base == null_seed(**kwargs)
    assert base != null_seed(**{**kwargs, "run_seed": 8})
    assert base != null_seed(**{**kwargs, "architecture": "bow_nuisance"})
    assert base != null_seed(**{**kwargs, "outer_fold": 2})
    assert base != null_seed(**{**kwargs, "inner_fold": 1})


def test_null_summary_reports_mean_p95_and_add_one_p_value():
    texts = _texts(seed=2)
    corpus = TrainingTextCorpus(texts)
    result = random_chunk_null(
        corpus=corpus,
        training_row_ids=range(len(texts)),
        evidence_texts=["ecog staged pain", "nothing here"],
        feature_tokens={"ecog_ps": frozenset({"ecog", "ps"})},
        draws=99,
        seed=3,
    )
    assert result.observed["ecog_ps"] == best_token_recall(
        ["ecog staged pain", "nothing here"], frozenset({"ecog", "ps"})
    ) == 0.5
    null = result.null_values["ecog_ps"]
    assert null.shape == (99,)
    summary = summarize_null(result.observed["ecog_ps"], null)
    assert summary["null_mean"] == pytest.approx(float(np.mean(null)))
    assert summary["null_p95"] == pytest.approx(float(np.quantile(null, 0.95)))
    assert summary["empirical_p_value"] == pytest.approx((1 + np.sum(null >= 0.5)) / 100)


def _run(tmp_path: Path) -> tuple[Path, Path]:
    run_dir = tmp_path / "run"
    rows = 40
    rng = np.random.default_rng(5)
    texts = [" ".join(rng.choice(["note", "routine", "visit", "ecog", "pain"], size=30))
             for _ in range(rows)]
    dataset = pd.DataFrame(
        {
            "clinical_text": texts,
            "treatment_indicator": [0, 1] * (rows // 2),
            "outcome_indicator": [0, 0, 1, 1] * (rows // 4),
            "true_ecog_ps": rng.integers(0, 3, size=rows),
        }
    )
    dataset_path = tmp_path / "dataset.parquet"
    dataset.to_parquet(dataset_path, index=False)
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"features": [{"name": "ecog_ps"}]}), encoding="utf-8")
    run_dir.mkdir()
    (run_dir / "run_config.json").write_text(
        json.dumps({"dataset": str(dataset_path), "seed": 11}), encoding="utf-8"
    )
    split = run_dir / "components" / "tfidf" / "split_provenance.jsonl"
    split.parent.mkdir(parents=True)
    split.write_text(
        json.dumps(
            {
                "outer_fold": 1,
                "fit_row_ids": list(range(30)),
                "heldout_row_ids": list(range(30, 40)),
                "inner_splits": [
                    {"inner_fold": 1, "fit_row_ids": list(range(15)),
                     "heldout_row_ids": list(range(15, 30))}
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    evidence_path = run_dir / "components" / "text_models" / "evidence.jsonl"
    evidence_path.parent.mkdir(parents=True)
    raw_rows = [
        {
            "source": "text_models",
            "outer_fold": 1,
            "inner_fold": inner,
            "scope": scope,
            "evidence": {
                "importance": {
                    "views": [
                        {
                            "view_name": "word_linear",
                            "treatment_positive": [
                                {"feature": term, "score": 1.0} for term in terms
                            ],
                            "outcome_positive": [],
                            "pseudo_target_positive": [],
                        }
                    ]
                }
            },
        }
        for inner, scope, terms in (
            (None, "full_outer_train", ["ecog ps", "routine visit"]),
            (1, "candidate_consistency_inner_train", ["pain"]),
        )
    ]
    evidence_path.write_text(
        "".join(json.dumps(row) + "\n" for row in raw_rows), encoding="utf-8"
    )
    materialize_stage1_architecture_artifacts(
        output_dir=run_dir,
        raw_handoff_rows=raw_rows,
        selected_architectures=("bow_nuisance",),
        source_artifacts={"text_models": evidence_path},
        selection_mode="explicit",
    )
    return run_dir, metadata


def test_evaluation_writes_per_context_null_comparisons(tmp_path):
    run_dir, metadata = _run(tmp_path)

    result = evaluate_stage1_architectures(run_dir=run_dir, metadata_path=metadata, null_draws=50)
    output = Path(result["output_dir"])
    table = pd.read_csv(output / "lexical_recovery_null.csv")

    assert output == run_dir / "evaluation" / "stage1"
    assert sorted(table["inner_fold"].fillna(0).astype(int).tolist()) == [0, 1]
    outer = table[table["inner_fold"].isna()].iloc[0]
    assert outer["observed"] == pytest.approx(1.0)  # "ecog ps" covers both tokens
    assert outer["draws"] == 50
    assert outer["evidence_items"] == 2
    assert 0.0 < outer["empirical_p_value"] <= 1.0
    inner = table[table["inner_fold"] == 1].iloc[0]
    assert inner["observed"] == pytest.approx(0.0)
    assert inner["empirical_p_value"] == pytest.approx(1.0)

    repeat = evaluate_stage1_architectures(
        run_dir=run_dir, metadata_path=metadata, null_draws=50, output_dir=tmp_path / "again"
    )
    pd.testing.assert_frame_equal(
        table, pd.read_csv(Path(repeat["output_dir"]) / "lexical_recovery_null.csv")
    )
    metrics = [
        json.loads(line)
        for line in (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    names = {row["metric"] for row in metrics if row["metric_family"] == "oracle_feature_recovery_null"}
    assert names == {
        "best_lexical_token_recall_observed",
        "best_lexical_token_recall_null_mean",
        "best_lexical_token_recall_null_p95",
        "best_lexical_token_recall_empirical_p_value",
    }
    manifest = json.loads((output / "evaluation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["consumed_split_provenance_sha256"]
    assert manifest["stage1_artifacts_frozen_before_oracle_load"] is True


def test_null_draws_zero_disables_the_null(tmp_path):
    run_dir, metadata = _run(tmp_path)

    result = evaluate_stage1_architectures(run_dir=run_dir, metadata_path=metadata, null_draws=0)
    table = pd.read_csv(Path(result["output_dir"]) / "lexical_recovery_null.csv")

    assert table["empirical_p_value"].isna().all()
    assert set(table["reason"]) == {"random-chunk null draws are disabled (--null-draws 0)"}
    with pytest.raises(ValueError, match="nonnegative"):
        evaluate_stage1_architectures(run_dir=run_dir, metadata_path=metadata, null_draws=-1)


def test_split_and_timing_inputs_are_frozen_before_the_oracle_dataset_loads(
    tmp_path, monkeypatch
):
    from oci.evaluation import stage1 as stage1_evaluation

    run_dir, metadata = _run(tmp_path)
    split = run_dir / "components" / "tfidf" / "split_provenance.jsonl"
    original_split = split.read_text(encoding="utf-8")
    timing = run_dir / "components" / "text_models" / "timing.jsonl"
    timing.write_text(
        json.dumps(
            {
                "schema_version": "stage1_architecture_timing_v1",
                "component": "text_models",
                "scope_id": "outer_001_full",
                "outer_fold": 1,
                "inner_fold": None,
                "status": "measured",
                "wall_seconds": 2.0,
                "entries": [
                    {"unit": "bow_importance_nuisance_terms", "attribution": "bow_nuisance",
                     "consumers": ["bow_nuisance"], "seconds": 2.0}
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    original_load = stage1_evaluation._load_dataset

    def load_after_tampering(path):
        # Anything read after this point would see the tampered files.
        split.write_text(original_split.replace('"fit_row_ids": [0', '"fit_row_ids": [39'),
                         encoding="utf-8")
        timing.write_text("", encoding="utf-8")
        return original_load(path)

    baseline = evaluate_stage1_architectures(
        run_dir=run_dir, metadata_path=metadata, null_draws=20, output_dir=tmp_path / "a"
    )
    split.write_text(original_split, encoding="utf-8")
    timing_text = json.dumps(baseline["timing"])
    monkeypatch.setattr(stage1_evaluation, "_load_dataset", load_after_tampering)
    tampered = evaluate_stage1_architectures(
        run_dir=run_dir, metadata_path=metadata, null_draws=20, output_dir=tmp_path / "b"
    )

    assert json.dumps(tampered["timing"]) == timing_text
    assert tampered["timing"]["architectures"]["bow_nuisance"]["exclusive_seconds"] == 2.0
    pd.testing.assert_frame_equal(
        pd.read_csv(tmp_path / "a" / "lexical_recovery_null.csv"),
        pd.read_csv(tmp_path / "b" / "lexical_recovery_null.csv"),
    )
