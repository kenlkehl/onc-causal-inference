from __future__ import annotations


import numpy as np
import pandas as pd

from oci.inference.stage2_sequential_consolidation import (
    SCHEMA_VERSION,
    Stage2SequentialConsolidationConfig,
    consolidate_stage2_candidates,
)


def _continuous(feature_id: str, name: str) -> dict[str, object]:
    return {
        "feature_id": feature_id,
        "name": name,
        "display_name": name.replace("_", " ").title(),
        "description": f"Pretreatment {name} measurement.",
        "value_type": "continuous",
        "modeling_strategy": "continuous",
        "categories_or_unit": ["unitless"],
        "measurement_definition": f"Use the recorded pretreatment {name}.",
        "missing_value_rule": "Null when absent.",
        "roles": [],
    }


def _embedding(texts, _model, _device):
    rows = []
    for text in texts:
        normalized = text.lower()
        if "third" in normalized:
            rows.append([0.0, 1.0])
        elif "histology" in normalized:
            rows.append([1.0, 0.0])
        else:
            rows.append([1.0, 0.0])
    return np.asarray(rows, dtype=float)


def test_created_latent_replaces_components_in_later_retrieval(tmp_path, monkeypatch):
    definitions = [
        _continuous("a", "first_burden"),
        _continuous("b", "second_burden"),
        _continuous("c", "third_marker"),
    ]
    frame = pd.DataFrame(
        {
            "_oci_row_id": np.arange(20),
            "first_burden": np.arange(20, dtype=float),
            "second_burden": np.arange(20, dtype=float),
            "third_marker": np.linspace(1.0, 2.0, 20),
        }
    )
    requests = []
    from oci.inference import stage2_sequential_consolidation as module
    original = module._decision_messages
    def capture(step_input, **kwargs):
        requests.append(step_input)
        return original(step_input, **kwargs)
    monkeypatch.setattr(module, "_decision_messages", capture)

    def request_json(messages, validate, *, request_kind="interpretation", **repair_kwargs):
        assert request_kind == "interpretation"
        assert "feature_id" not in messages[1]["content"]
        assert "Paired measurements" in messages[1]["content"]
        assert repair_kwargs["conservative_validation_fallback"]["action"] == "leave_unchanged"
        assert repair_kwargs["fallback_after_same_error"] == 3
        if len(requests) == 1:
            return validate({"action": "merge", "reason": "Equivalent values and units.",
                "members": ["First Burden", "Second Burden"],
                "canonical_label": "Baseline burden", "category_equivalences": []})
        return validate({"action": "keep", "reason": "Distinct clinical measurements."})

    consolidated, active, report, entries = consolidate_stage2_candidates(
        extracted_fit=frame,
        definitions=definitions,
        request_json=request_json,
        policy=Stage2SequentialConsolidationConfig(
            enabled=True,
            neighbor_count=1,
            embedding_model="test-model",
            embedding_device="cpu",
        ),
        output_dir=tmp_path / "consolidation",
        request_model="test-llm",
        embedding_function=_embedding,
    )

    assert report["schema_version"] == SCHEMA_VERSION
    assert report["latents_created"] == 1
    assert report["components_consumed"] == 2
    assert len(entries) == 1
    latent_id = entries[0]["latent_id"]
    assert [feature["feature_id"] for feature in active] == ["c", latent_id]
    assert entries[0]["name"] in consolidated
    assert len(requests) == 2
    second_cluster_ids = {
        feature["feature_id"] for feature in requests[1]["features"]
    }
    assert second_cluster_ids == {"c", latent_id}
    assert "a" not in second_cluster_ids
    assert "b" not in second_cluster_ids

    def unexpected_request(*_args, **_kwargs):
        raise AssertionError("completed consolidation should be reconstructed from checkpoint")

    resumed, resumed_active, resumed_report, resumed_entries = (
        consolidate_stage2_candidates(
            extracted_fit=frame,
            definitions=definitions,
            request_json=unexpected_request,
            policy=Stage2SequentialConsolidationConfig(
                enabled=True,
                neighbor_count=1,
                embedding_model="test-model",
                embedding_device="cpu",
            ),
            output_dir=tmp_path / "consolidation",
            request_model="test-llm",
            embedding_function=_embedding,
        )
    )
    assert resumed_report == report
    assert resumed_entries == entries
    assert [feature["feature_id"] for feature in resumed_active] == ["c", latent_id]
    np.testing.assert_allclose(
        resumed[entries[0]["name"]],
        consolidated[entries[0]["name"]],
        equal_nan=True,
    )
