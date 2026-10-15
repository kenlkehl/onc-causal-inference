"""The run seed reaches every TF-IDF random draw (S1-25)."""

from __future__ import annotations

import pytest

from oci.inference import tfidf_topic_stage1
from oci.inference.tfidf_topic_stage1 import run_tfidf_topic_stage1, seeded_tfidf_topic_config
from tests.test_tfidf_topic_stage1_end_to_end import _config, _dataset


class _Captured(Exception):
    pass


def _context_topic_config(tmp_path, monkeypatch, *, run_seed: int, orphan: bool = False):
    config = _config(orphan=orphan)
    config.seed = run_seed
    captured = []

    def capture(**kwargs):
        captured.append(kwargs["config"])
        raise _Captured()

    monkeypatch.setattr(tfidf_topic_stage1, "fit_tfidf_topic_context", capture)
    with pytest.raises(Exception):
        run_tfidf_topic_stage1(
            dataset=_dataset(),
            config=config,
            output_path=tmp_path / "predictions.parquet",
            artifact_dir=tmp_path,
            handoff_path=tmp_path / "evidence.jsonl",
        )
    assert captured, "the TF-IDF context fit was never reached"
    return captured[0]


@pytest.mark.parametrize("orphan", [False, True])
def test_tfidf_context_fits_draw_their_seeds_from_the_run_seed(tmp_path, monkeypatch, orphan):
    first = _context_topic_config(tmp_path / "a", monkeypatch, run_seed=11, orphan=orphan)
    repeat = _context_topic_config(tmp_path / "b", monkeypatch, run_seed=11, orphan=orphan)
    other = _context_topic_config(tmp_path / "c", monkeypatch, run_seed=12, orphan=orphan)

    assert (first.random_state, first.topic_seeds) == (repeat.random_state, repeat.topic_seeds)
    assert first.random_state != other.random_state
    assert not set(first.topic_seeds) & set(other.topic_seeds)
    assert len(set(first.topic_seeds)) == len(first.topic_seeds) == 3
    # The configured values are stream selectors, not the seeds themselves.
    assert first.random_state != 42
    assert first.topic_seeds != [42, 43, 44]


def test_seeded_config_keeps_every_non_seed_setting():
    config = _config(orphan=True)
    seeded = seeded_tfidf_topic_config(config)
    original = config.architecture.multi_model_forest.tfidf_topic

    assert seeded is not original
    assert original.random_state == 42 and original.topic_seeds == [42, 43, 44]
    assert seeded.topic_count == original.topic_count
    assert seeded.orphan_ngram_enabled is True
    assert seeded.vectorizer_scientific == original.vectorizer_scientific
    assert 0 <= seeded.random_state < 2**30
