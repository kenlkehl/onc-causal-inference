import hashlib
import json

import numpy as np
import pytest

from oci.inference.stage2_candidate_consolidation import (
    policy_from_mapping,
)
from oci.inference import plain_handoff_stage2 as workflow


def entries(count):
    return [
        {
            "entry_id": f"d{index:04d}_01",
            "definition": {
                "name": f"variable_{index:03d}", "description": f"Measurement {index}.",
                "value_type": "continuous", "unit": "mg/dL", "categories": [], "label_definitions": {},
                "measurement_definition": "Use the latest value.", "missing_value_rule": "Null when absent.",
                "conflict_resolution": {"strategy": "latest", "positive_category": None},
            },
            "locked": False, "explicit_feature": None, "origin_variable_ids": [f"d{index:04d}_01"],
            "origin_candidate_ids": [f"candidate_{index:03d}"], "supporting_packet_ids": [],
            "supporting_architectures": ["test"], "label_map": {},
        }
        for index in range(1, count + 1)
    ]


def encoder(texts, _model, _device):
    return np.array([[1, *hashlib.sha256(text.encode()).digest()[:7]] for text in texts], dtype=float)


def runner(**kwargs):
    return workflow.PlainHandoffStage2(
        config=workflow.PlainHandoffStage2Config(endpoint="http://test/v1", model="test-model",
                                               workers=2, **kwargs),
        clinical_question="Not used for consolidation",
        completion=lambda *_: json.dumps({"groups": []}),
    )


def test_harmonized_merge_stops_resumes_and_preserves_origins(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, "_encode_candidate_selection_texts", encoder)
    model = runner()
    source = entries(45)
    result = model._harmonize_variable_pool(outer_fold=1, entries=source, output_dir=tmp_path)
    assert sorted(entry["entry_id"] for entry in result) == sorted(entry["entry_id"] for entry in source)
    complete = json.loads((tmp_path / "complete.json").read_text())
    assert complete["stopped_reason"] == "diminishing_returns"
    assert complete["rounds_executed"] == 3
    assert not (tmp_path / "round_004").exists()
    model.completion = lambda *_: pytest.fail("A saved LLM request was repeated")
    monkeypatch.setattr(workflow, "_encode_candidate_selection_texts",
                        lambda *_: pytest.fail("Saved embeddings were recomputed"))
    assert model._harmonize_variable_pool(outer_fold=1, entries=source, output_dir=tmp_path) == result


def test_exact_duplicates_merge_without_a_model_review(tmp_path):
    model = runner()
    model.completion = lambda *_: pytest.fail("Exact duplicates need no model review")
    pair = entries(2)
    pair[1]["definition"] = dict(pair[0]["definition"], description="Another description.")
    result = model._harmonize_variable_pool(outer_fold=1, entries=pair, output_dir=tmp_path)
    [merged] = result
    assert merged["origin_candidate_ids"] == ["candidate_001", "candidate_002"]
    complete = json.loads((tmp_path / "complete.json").read_text())
    assert complete["stopped_reason"] == "nothing_to_merge"


@pytest.mark.parametrize("settings", [{"unknown": True}])
def test_invalid_policy_is_rejected(settings):
    with pytest.raises(ValueError):
        policy_from_mapping(settings)
