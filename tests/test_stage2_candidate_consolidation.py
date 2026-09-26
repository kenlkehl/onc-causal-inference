import hashlib
import json

import numpy as np
import pytest

from oci.inference.stage2_candidate_consolidation import (
    CandidateConsolidationPolicy, CandidateEmbeddingCache, mixed_batches,
    diminishing_returns, policy_from_mapping,
)
from oci.inference import plain_handoff_stage2 as workflow


def groups(count):
    return workflow._materialize_exact_name_groups([
        {"candidate_id": f"candidate_{i:03d}", "name": f"variable_{i:03d}",
         "description": f"Measurement {i}.", "architecture": "test",
         "supporting_packet_ids": [], "evidence_axes": ["outcome"]}
        for i in range(count)
    ])


def encoder(texts, _model, _device):
    return np.array([[1, *hashlib.sha256(text.encode()).digest()[:7]] for text in texts], dtype=float)


def runner(**kwargs):
    return workflow.PlainHandoffStage2(
        config=workflow.PlainHandoffStage2Config(endpoint="http://test/v1", model="test-model",
                                               workers=2, **kwargs),
        clinical_question="Not used for consolidation",
        completion=lambda *_: json.dumps({"merge_directives": []}),
    )


def test_mixed_round_covers_all_candidates_once_with_little_random_and_changes_next_round():
    source = groups(200)
    matrix = encoder([item["name"] for item in source], "", "")
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    args = dict(batch_size=20, seed=41, policy=CandidateConsolidationPolicy(), embeddings=matrix)
    first, methods = mixed_batches(source, round_number=1, **args)
    second, _ = mixed_batches(source, round_number=2, **args)
    assert methods.count("semantic") == 6
    assert methods.count("alphabetical") == 3
    assert methods.count("random") == 1
    assert sorted(item["name"] for batch in first for item in batch) == [item["name"] for item in source]
    assert all(len(batch) == 20 for batch in first)
    assert first != second
    assert (first, methods) == mixed_batches(source, round_number=1, **args)


def test_semantic_batch_brings_distant_aliases_together_without_merging_them():
    source = groups(8)
    matrix = np.eye(8)
    matrix[-1] = matrix[0]
    batches, methods = mixed_batches(source, batch_size=2, round_number=1, seed=0,
                                    policy=CandidateConsolidationPolicy(), embeddings=matrix)
    assert methods[0] == "semantic"
    assert [item["name"] for item in batches[0]] == ["variable_000", "variable_007"]
    assert sum(map(len, batches)) == len(source)


def test_embedding_cache_reuses_vectors_and_only_embeds_changed_text(tmp_path):
    calls = []
    def encode(texts, model, device):
        calls.append(list(texts))
        return encoder(texts, model, device)
    path = tmp_path / "vectors.npz"
    cache = CandidateEmbeddingCache(path, CandidateConsolidationPolicy(), encode)
    original = cache.matrix(["hemoglobin", "age"])
    np.testing.assert_array_equal(original, cache.matrix(["hemoglobin", "age"]))
    restored = CandidateEmbeddingCache(path, CandidateConsolidationPolicy(), encode)
    np.testing.assert_array_equal(original, restored.matrix(["hemoglobin", "age"]))
    restored.matrix(["hemoglobin", "new merged description"])
    assert calls == [["hemoglobin", "age"], ["new merged description"]]


def test_low_yield_requires_consecutive_successful_rounds_and_minimum_exposure():
    policy = CandidateConsolidationPolicy()
    def row(before, after, failures=0):
        return dict(input_groups=before, output_groups=after, validation_fallback_batches=failures)
    assert not diminishing_returns([row(1000, 998), row(998, 996)], policy)
    assert diminishing_returns([row(1100, 1000), row(1000, 998), row(998, 996)], policy)
    assert not diminishing_returns([row(1100, 1000), row(1000, 998, 1), row(998, 996)], policy)
    assert not diminishing_returns([row(1100, 1000), row(1000, 980), row(980, 978)], policy)
    assert not diminishing_returns([row(1100, 1000), row(1000, 995), row(995, 993)], policy)


def test_mixed_workflow_stops_resumes_and_preserves_origins(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, "_encode_candidate_selection_texts", encoder)
    model = runner()
    source = groups(45)
    result = model._consolidate_candidate_pool(outer_fold=1, groups=source, output_dir=tmp_path)
    assert result == source
    complete = json.loads((tmp_path / "complete.json").read_text())
    assert complete["stopped_reason"] == "diminishing_returns"
    assert complete["rounds_executed"] == 3
    assert not (tmp_path / "round_004").exists()
    model.completion = lambda *_: pytest.fail("A saved LLM request was repeated")
    monkeypatch.setattr(workflow, "_encode_candidate_selection_texts",
                        lambda *_: pytest.fail("Saved embeddings were recomputed"))
    assert model._consolidate_candidate_pool(outer_fold=1, groups=source, output_dir=tmp_path) == result


def test_failed_review_does_not_count_as_convergence(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, "_encode_candidate_selection_texts", encoder)
    def request(**kwargs):
        if kwargs["input_value"]["round"] == 2:
            raise workflow.Stage2ResponseValidationError("malformed response")
        return {"merge_directives": []}
    monkeypatch.setattr(workflow, "_checkpointed_request_json", request)
    model = runner(consolidation_max_rounds=5)
    assert len(model._consolidate_candidate_pool(outer_fold=1, groups=groups(41), output_dir=tmp_path)) == 41
    complete = json.loads((tmp_path / "complete.json").read_text())
    assert complete["rounds_executed"] == 4
    assert complete["stopped_reason"] == "diminishing_returns"
    assert complete["validation_fallback_batches"] == 3


@pytest.mark.parametrize("settings", [
    {"semantic_fraction": .95, "random_fraction": .1}, {"random_fraction": float("nan")},
    {"early_stop_patience": 0}, {"early_stop_min_reduction": -1},
    {"unknown": True},
])
def test_invalid_policy_is_rejected(settings):
    with pytest.raises(ValueError):
        policy_from_mapping(settings)
