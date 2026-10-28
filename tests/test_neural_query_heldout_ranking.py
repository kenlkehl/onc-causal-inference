"""Neural-query consensus groups are ranked by held-out |z| (S1-24)."""

from __future__ import annotations

import json
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
    build_query_evidence,
)
from oci.inference.neural_query_evidence_contract import (
    NEURAL_QUERY_CONTEXT_SCHEMA_VERSION,
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


def test_records_carry_both_scores_and_heldout_p_values():
    bank = _bank(_candidates([9.0, 1.0], [-0.5, 2.0]), groups=2)

    first = bank["records"][0]
    assert _group_of(first) == 1
    assert first["median_abs_heldout_z"] == pytest.approx(2.0)
    assert first["median_abs_train_score"] == pytest.approx(1.0)
    assert first["median_heldout_p_value"] == pytest.approx(np.exp(-2.0))
    assert first["heldout_member_count"] == 2
    assert [
        member["validation_audit_two_sided_p_value"] for member in first["members"]
    ] == [pytest.approx(np.exp(-2.0))] * 2


def test_heldout_ties_break_on_training_score_then_medoid_id():
    bank = _bank(_candidates([2.0, 7.0, 7.0], [1.0, 1.0, 1.0]), groups=3)

    order = [_group_of(record) for record in bank["records"]]
    # Equal held-out |z|: higher training score first, then the medoid ID.
    assert order == [1, 2, 0]


def test_groups_without_heldout_statistics_rank_last_and_are_counted():
    bank = _bank(_candidates([9.0, 8.0, 1.0], [None, None, 0.1]), groups=3)

    order = [_group_of(record) for record in bank["records"]]
    # The only group with held-out statistics leads despite its training
    # score; the others follow in training-score order.
    assert order == [2, 0, 1]
    assert bank["ranking"]["groups_without_heldout_statistics"] == 2
    assert bank["ranking"]["groups_with_heldout_statistics"] == 1
    assert bank["records"][1]["median_abs_heldout_z"] is None
    assert bank["records"][1]["ranking_score_source"].startswith("median_abs_train_score")


def test_query_evidence_payload_records_both_scores_and_member_p_values():
    bank = _bank(_candidates([9.0, 1.0], [-0.5, 2.0]), groups=2)
    records = bank["records"]
    config = NeuralQueryAgenticForestConfig(
        evidence_top_patients=1,
        evidence_background_patients=None,
        evidence_top_ngrams=None,
        evidence_ngram_stop_words=None,
        evidence_ngram_range_min=1,
        evidence_ngram_range_max=1,
    )
    evidence = build_query_evidence(
        bank="effect",
        queries=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        query_records=records,
        row_ids=[0, 1],
        chunk_matrices=[
            np.array([[1.0, 0.0]], dtype=np.float32),
            np.array([[0.0, 1.0]], dtype=np.float32),
        ],
        all_chunk_texts=[["alpha signal"], ["beta signal"]],
        config=config,
        device="cpu",
        seed=7,
    )

    first = evidence[0]
    assert first["rank"] == 1
    assert first["ranking"]["version"] == NEURAL_QUERY_RANKING_VERSION
    assert first["median_abs_heldout_z"] == pytest.approx(2.0)
    assert first["median_abs_train_score"] == pytest.approx(1.0)
    assert first["median_heldout_p_value"] == pytest.approx(np.exp(-2.0))
    assert [item["heldout_p_value"] for item in first["member_heldout_statistics"]] == [
        pytest.approx(np.exp(-2.0)),
        pytest.approx(np.exp(-2.0)),
    ]
    assert {item["train_score"] for item in first["member_heldout_statistics"]} == {1.0}


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


def test_contexts_written_before_heldout_ranking_are_refit(tmp_path, monkeypatch):
    import joblib

    specs = [
        {"scope_id": "outer_001_full", "outer_fold": 1},
        {"scope_id": "outer_002_full", "outer_fold": 2},
    ]
    for spec, schema in zip(specs, (None, NEURAL_QUERY_CONTEXT_SCHEMA_VERSION)):
        context_dir = tmp_path / spec["scope_id"]
        context_dir.mkdir()
        marker = {"status": "complete"}
        if schema is not None:
            marker["schema_version"] = schema
        (context_dir / "complete.json").write_text(json.dumps(marker), encoding="utf-8")
        (context_dir / "evidence.json").write_text(
            json.dumps({"outer_fold": spec["outer_fold"], "evidence": []}),
            encoding="utf-8",
        )
    refit: list[str] = []

    def run_lane(**kwargs):
        refit.extend(spec["scope_id"] for spec in kwargs["specs"])
        return [{"outer_fold": spec["outer_fold"], "evidence": []} for spec in kwargs["specs"]]

    monkeypatch.setattr(workflow, "_stage1_context_specs", lambda _context: specs)
    monkeypatch.setattr(workflow, "_run_neural_query_context_lane", run_lane)
    monkeypatch.setattr(
        joblib,
        "Parallel",
        lambda **_: lambda tasks: [fn(*args, **kwargs) for fn, args, kwargs in tasks],
    )
    context = SimpleNamespace(
        config=SimpleNamespace(devices=("cpu",), workers=1, contexts_per_gpu=1),
        dataset=None,
        applied_config=None,
        neural_query_config=None,
    )

    workflow._neural_queries_component(context, tmp_path)

    assert refit == ["outer_001_full"]


def test_context_sidecars_and_payload_record_ranking_scores_and_timing(tmp_path, monkeypatch):
    bank_records = _bank(_candidates([9.0, 1.0], [-0.5, 2.0]), groups=2)["records"]

    def discover(**kwargs):
        rows = len(kwargs["row_ids"])
        return {
            "banks": {
                bank: {
                    "queries": np.eye(2, dtype=np.float32),
                    "train_activations": np.zeros((rows, 2), dtype=np.float32),
                    "records": [dict(record, bank=bank) for record in bank_records],
                    "consensus": {"ranking": {"version": NEURAL_QUERY_RANKING_VERSION,
                                              "groups_without_heldout_statistics": 0}},
                }
                for bank in runtime.BANKS
            }
        }

    class Generator:
        def chunk_matrices(self, rows):
            return [np.eye(2, dtype=np.float32) for _ in rows]

        def chunk_texts(self, rows):
            return [["alpha signal", "beta signal"] for _ in rows]

    monkeypatch.setattr(runtime, "fit_context_query_discovery", discover)
    dataset = pd.DataFrame(
        {"text": [f"note {i}" for i in range(8)], "t": [0, 1] * 4, "y": [1, 0] * 4}
    )
    fold_dir = tmp_path / "outer_001_full"
    evidence = workflow._run_one_neural_query_context(
        dataset=dataset,
        config=SimpleNamespace(
            text_column="text", treatment_column="t", outcome_column="y",
            outcome_type="binary", seed=1,
        ),
        applied_config=SimpleNamespace(
            architecture=SimpleNamespace(
                multi_model_forest=SimpleNamespace(
                    bow_views=[], nuisance_folds=2,
                    tfidf_topic=SimpleNamespace(
                        nuisance_stack_scientific=TfidfNuisanceStackScientificConfig()
                    ),
                )
            )
        ),
        neural_query_config=NeuralQueryAgenticForestConfig(
            evidence_top_patients=1, evidence_background_patients=None,
            evidence_top_ngrams=None, evidence_ngram_stop_words=None,
        ),
        generator=Generator(),
        spec={"scope_id": "outer_001_full", "outer_fold": 1, "fold_key": 1,
              "scope": "full_outer_train", "train_idx": list(range(6)),
              "heldout_idx": [6, 7]},
        fold_dir=fold_dir,
        device="cpu",
    )

    assert evidence["schema_version"] == NEURAL_QUERY_CONTEXT_SCHEMA_VERSION
    assert evidence["query_ranking"]["version"] == NEURAL_QUERY_RANKING_VERSION
    first = evidence["evidence"][0]
    assert (first["median_abs_heldout_z"], first["median_abs_train_score"]) == (
        pytest.approx(2.0), pytest.approx(1.0)
    )
    sidecar = json.loads((fold_dir / "query_records.json").read_text(encoding="utf-8"))
    assert sidecar["schema_version"] == NEURAL_QUERY_CONTEXT_SCHEMA_VERSION
    assert set(sidecar["banks"]) == set(runtime.BANKS)
    record = sidecar["banks"]["effect"][0]
    assert record["median_abs_heldout_z"] == pytest.approx(2.0)
    assert record["median_abs_train_score"] == pytest.approx(1.0)
    assert record["median_heldout_p_value"] == pytest.approx(np.exp(-2.0))
    marker = json.loads((fold_dir / "complete.json").read_text(encoding="utf-8"))
    assert marker["schema_version"] == NEURAL_QUERY_CONTEXT_SCHEMA_VERSION
    timing = json.loads((fold_dir / "timing.json").read_text(encoding="utf-8"))
    assert timing["status"] == "measured"
    assert {entry["attribution"] for entry in timing["entries"]} >= {"neural_query_moments"}
