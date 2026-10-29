"""Neural-query consensus groups are ranked by held-out |z| (S1-24)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from oci.config import TfidfNuisanceStackScientificConfig
from oci.inference import neural_query_discovery_runtime as runtime
from oci.inference import research_all_evidence_workflow as workflow
from oci.inference.neural_cohort_witness import (
    NeuralCohortWitnessConfig,
    build_ungated_consensus_query_bank,
)
from oci.inference.neural_query_agentic_forest import (
    NeuralQueryAgenticForestConfig,
)
from oci.inference.neural_query_evidence_contract import (
    NEURAL_QUERY_RANKING_VERSION,
)


def _patterns(groups: int, members: int, patients: int = 60) -> np.ndarray:
    """Activation columns: ``members`` near-identical columns per group."""

    rng = np.random.default_rng(3)
    base = rng.normal(size=(patients, groups))
    columns = [
        base[:, group] + 1e-3 * rng.normal(size=patients)
        for group in range(groups)
        for _member in range(members)
    ]
    return np.column_stack(columns)


def _candidates(
    train: list[float],
    heldout: list[float | None],
    *,
    members: int = 2,
) -> list[dict]:
    output = []
    for group, (train_score, heldout_z) in enumerate(zip(train, heldout)):
        for member in range(members):
            candidate = {
                "candidate_id": f"effect_fold_{member + 1:02d}_query_{group + 1:03d}",
                "bank": "effect",
                "subfold": member + 1,
                "query": np.eye(4, dtype=np.float32)[group % 4],
                "train_standardized_score": float(train_score),
            }
            if heldout_z is not None:
                candidate["validation_audit_standardized_score"] = float(heldout_z)
                candidate["validation_audit_two_sided_p_value"] = float(
                    np.exp(-abs(heldout_z))
                )
            output.append(candidate)
    return output


def _bank(candidates: list[dict], groups: int):
    return build_ungated_consensus_query_bank(
        candidates,
        candidate_activations=_patterns(groups, len(candidates) // groups),
        n_queries=groups,
        bank="effect",
        seed=11,
        config=NeuralCohortWitnessConfig(n_prototypes=groups, initial_pool_size=groups),
    )


def _group_of(record: dict) -> int:
    return int(record["medoid_candidate_id"].rsplit("_", 1)[-1]) - 1


def test_groups_rank_by_median_heldout_abs_z_not_training_score():
    # Training scores rank the groups 0 > 1 > 2; held-out |z| ranks 2 > 0 > 1.
    bank = _bank(_candidates([9.0, 5.0, 1.0], [-1.5, 0.2, 3.0]), groups=3)

    order = [_group_of(record) for record in bank["records"]]
    assert order == [2, 0, 1]
    assert [record["rank"] for record in bank["records"]] == [1, 2, 3]
    assert [record["query_id"] for record in bank["records"]] == [
        "effect_query_001",
        "effect_query_002",
        "effect_query_003",
    ]
    # The returned initializers follow the same order as the records.
    for query, record in zip(bank["queries"], bank["records"]):
        assert np.argmax(query) == _group_of(record)
    assert all(
        record["ranking_score_source"] == "median_abs_heldout_z"
        for record in bank["records"]
    )
    assert bank["ranking"]["version"] == NEURAL_QUERY_RANKING_VERSION
    assert bank["ranking"]["groups_without_heldout_statistics"] == 0
    # No gating: every candidate stays assigned and the budget is unchanged.
    assert bank["selected_count"] == 3
    assert sum(record["member_count"] for record in bank["records"]) == 6


def test_validation_rows_are_inner_partitions_of_the_context_fit_rows(monkeypatch):
    captured: list[tuple[int, np.ndarray, np.ndarray]] = []

    def subfold(*, fold, train_indices, validation_indices, **_kwargs):
        captured.append((fold, np.asarray(train_indices), np.asarray(validation_indices)))
        return {"fold": fold, "banks": {bank: {"candidates": []} for bank in runtime.BANKS}}

    def final_bank(**kwargs):
        index = kwargs["bank_index"]
        return {
            "bank": kwargs["bank"],
            "bank_index": index,
            "result": {},
            "consensus_seed": kwargs["seed"] + 1000 + index,
            "final_refit_seed": kwargs["seed"] + 2000 + index,
        }

    monkeypatch.setattr(runtime, "_fit_subfold", subfold)
    monkeypatch.setattr(runtime, "_fit_final_bank", final_bank)
    n = 40
    runtime.fit_in_memory_query_discovery(
        fit_ids=tuple(range(100, 100 + n)),
        fit_chunks=[np.ones((1, 2), dtype=np.float32) for _ in range(n)],
        fit_texts=tuple(f"text {index}" for index in range(n)),
        treatment=np.asarray([index % 2 for index in range(n)], dtype=float),
        outcome=np.asarray([(index // 2) % 2 for index in range(n)], dtype=float),
        outcome_binary=True,
        fit_e=np.full(n, 0.5),
        fit_m=np.full(n, 0.5),
        fit_dina={"a": np.full(n, 0.5), "nu": np.zeros(n)},
        nuisance_views=[],
        nuisance_stack_config=TfidfNuisanceStackScientificConfig(),
        config=NeuralQueryAgenticForestConfig(query_inner_folds=4),
        nuisance_folds=2,
        devices=("cpu",),
        seed=5,
    )

    assert len(captured) == 4
    validation_union = np.concatenate([validation for _fold, _train, validation in captured])
    # Positions index the context's own fit rows only, and the validation
    # partitions cover them exactly once.
    assert sorted(validation_union.tolist()) == list(range(n))
    for _fold, train, validation in captured:
        assert not set(train.tolist()) & set(validation.tolist())
        assert sorted([*train.tolist(), *validation.tolist()]) == list(range(n))


def test_context_discovery_never_receives_the_context_heldout_rows(tmp_path, monkeypatch):
    class Generator:
        def chunk_matrices(self, rows):
            return [np.ones((1, 2), dtype=np.float32) for _ in rows]

        def chunk_texts(self, rows):
            return [["text"] for _ in rows]

    class Reached(Exception):
        pass

    seen = {}

    def discover(**kwargs):
        seen["rows"] = tuple(kwargs["row_ids"])
        raise Reached

    monkeypatch.setattr(runtime, "fit_context_query_discovery", discover)
    dataset = pd.DataFrame(
        {"text": [f"note {i}" for i in range(10)], "t": [0, 1] * 5, "y": [1, 0] * 5}
    )
    with pytest.raises(Reached):
        workflow._run_one_neural_query_context(
            dataset=dataset,
            config=SimpleNamespace(
                text_column="text",
                treatment_column="t",
                outcome_column="y",
                outcome_type="binary",
                seed=1,
            ),
            applied_config=SimpleNamespace(
                architecture=SimpleNamespace(
                    multi_model_forest=SimpleNamespace(
                        bow_views=[],
                        nuisance_folds=2,
                        tfidf_topic=SimpleNamespace(
                            nuisance_stack_scientific=TfidfNuisanceStackScientificConfig()
                        ),
                    )
                )
            ),
            neural_query_config=NeuralQueryAgenticForestConfig(),
            generator=Generator(),
            spec={
                "scope_id": "outer_001_full",
                "outer_fold": 1,
                "fold_key": 1,
                "scope": "full_outer_train",
                "train_idx": [0, 1, 2, 3, 4, 5, 6],
                "heldout_idx": [7, 8, 9],
            },
            fold_dir=tmp_path / "ctx",
            device="cpu",
        )
    assert seen["rows"] == (0, 1, 2, 3, 4, 5, 6)
