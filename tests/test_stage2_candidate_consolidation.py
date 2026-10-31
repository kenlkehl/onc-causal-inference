import hashlib
import json

import numpy as np
import pytest

from oci.inference.stage2_candidate_consolidation import (
    policy_from_mapping,
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


@pytest.mark.parametrize("settings", [{"unknown": True}])
def test_invalid_policy_is_rejected(settings):
    with pytest.raises(ValueError):
        policy_from_mapping(settings)
