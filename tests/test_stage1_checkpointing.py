from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from oci.inference import multi_model_forest_stage1, multi_model_pair_uplift
from oci.inference.stage1_checkpointing import (
    STAGE1_CHECKPOINT_SCHEMA_VERSION,
    Stage1CheckpointBusyError,
    Stage1CheckpointStore,
    dataset_file_identity,
    row_id_identity,
)


def _store(tmp_path: Path, *, identity: str = "context-a") -> Stage1CheckpointStore:
    return Stage1CheckpointStore(
        tmp_path / "checkpoints" / "v1",
        context_identity={
            "context": identity,
            "fit": row_id_identity([4, 2, 9]),
            "heldout": row_id_identity([1, 8]),
        },
        base_seed=123,
    )


class _FakeHTRNet:
    def __init__(self, *, extractor, **_kwargs):
        self.extractor = extractor

    def to(self, _device):
        return self


class _FakeHTRRunner:
    def __init__(self, *, interrupt_stage: str | None = None, interrupt_fold: int = -1):
        self.avf_config = SimpleNamespace(
            nuisance_folds=3,
            effect_folds=3,
            nuisance_calibration="none",
            effect_objective="pseudo_outcome_mse",
            e_clip=0.01,
            r_stage_min_propensity=0.0,
            r_stage_max_propensity=1.0,
        )
        self.config = SimpleNamespace(
            treatment_column="treatment",
            outcome_column="outcome",
            outcome_type="binary",
            architecture=SimpleNamespace(htr_prediction_head_hidden_dim=4),
        )
        self.device = torch.device("cpu")
        self.interrupt_stage = interrupt_stage
        self.interrupt_fold = int(interrupt_fold)
        self.nuisance_calls: list[int] = []
        self.effect_calls: list[int] = []

    def _create_extractor(self):
        return object()

    def _train_nuisance_model(self, _model, _df, _fit_pos, *, fold, **_kwargs):
        self.nuisance_calls.append(int(fold))
        if self.interrupt_stage == "nuisance" and int(fold) == self.interrupt_fold:
            raise RuntimeError("simulated HTR nuisance interruption")

    def _predict_nuisance_model(self, _model, frame):
        row_ids = frame["_oci_row_id"].to_numpy(dtype=float)
        return 0.2 + (row_ids % 5) / 10.0, 0.3 + (row_ids % 4) / 10.0

    def _train_effect_model(self, _model, _df, _fit_pos, *_args, fold, **_kwargs):
        self.effect_calls.append(int(fold))
        if self.interrupt_stage == "effect" and int(fold) == self.interrupt_fold:
            raise RuntimeError("simulated HTR effect interruption")

    def _predict_effect_model(self, _model, frame):
        return frame["_oci_row_id"].to_numpy(dtype=float) / 20.0

    def _attention_evidence(self, _extractor, frame, *, fold, stage, **_kwargs):
        return [
            {"row_id": int(row_id), "inner_fold": int(fold), "stage": str(stage)}
            for row_id in frame["_oci_row_id"].to_numpy()
        ]

    def _cleanup_model(self, _model):
        return None

    def _fold_n_jobs(self, _folds):
        return 3

    def _device_context_for_inner_fold(self, _fold):
        return nullcontext()


def _htr_provider(store: Stage1CheckpointStore, runner: _FakeHTRRunner):
    provider = object.__new__(multi_model_forest_stage1.MultiModelForestStage1HTRProvider)
    provider._runner = runner
    provider.config = runner.config
    provider.checkpoint_store = store
    provider.native_capture_sink = None
    provider.native_pair_capture_sink = None
    provider._ensure_runner = lambda _frame: runner
    return provider


def test_checkpoint_round_trip_reuses_without_byte_hashing(tmp_path):
    store = _store(tmp_path)
    calls = 0

    def compute():
        nonlocal calls
        calls += 1
        return {
            "array": np.asarray([1.5, np.nan, 3.5]),
            "frame": pd.DataFrame({"row_id": [4, 2], "score": [0.2, 0.8]}),
            "nested": [{"value": np.float64(7.0)}],
        }

    first = store.run(
        "htr/nuisance/fold_001",
        compute,
        parameters={"fold": 1},
        rows={"validation": row_id_identity([4, 2])},
    )
    second = store.run(
        "htr/nuisance/fold_001",
        compute,
        parameters={"fold": 1},
        rows={"validation": row_id_identity([4, 2])},
    )

    assert calls == 1
    assert first.reused is False
    assert second.reused is True
    np.testing.assert_equal(second.value["array"], first.value["array"])
    pd.testing.assert_frame_equal(second.value["frame"], first.value["frame"])
    marker = next((store.root / "units").rglob("complete.json"))
    marker_value = json.loads(marker.read_text(encoding="utf-8"))
    assert marker_value["schema_version"] == STAGE1_CHECKPOINT_SCHEMA_VERSION
    assert marker_value["content_hashing"] is False
    assert marker_value["validation"] == "marker_identity_row_ids_file_presence_and_size"


def test_completed_leaf_survives_interruption_and_resumes_independently(tmp_path):
    store = _store(tmp_path)
    completed_calls = 0

    def completed_compute():
        nonlocal completed_calls
        completed_calls += 1
        return np.asarray([10.0, 20.0])

    store.run("bow/nuisance/fold_001", completed_compute)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        store.run(
            "bow/nuisance/fold_002",
            lambda: (_ for _ in ()).throw(RuntimeError("simulated interruption")),
        )

    resumed = _store(tmp_path)
    reused = resumed.run("bow/nuisance/fold_001", completed_compute)
    second = resumed.run(
        "bow/nuisance/fold_002",
        lambda: np.asarray([30.0, 40.0]),
    )

    assert completed_calls == 1
    assert reused.reused is True
    assert second.reused is False
    progress = json.loads(
        (resumed.root / "checkpoint_progress.json").read_text(encoding="utf-8")
    )
    assert progress["units"]["bow/nuisance/fold_001"]["status"] == "reused"
    assert progress["units"]["bow/nuisance/fold_002"]["status"] == "completed"


def test_truncated_payload_is_recomputed_instead_of_adopted(tmp_path):
    store = _store(tmp_path)
    first = store.run("embedding/contrast", lambda: np.arange(6, dtype=float))
    checkpoint_dir = store.root / "units" / "embedding" / "contrast" / first.fingerprint
    array_path = next(checkpoint_dir.glob("array_*.npy"))
    array_path.write_bytes(b"truncated")
    calls = 0

    def recompute():
        nonlocal calls
        calls += 1
        return np.arange(6, dtype=float) + 100

    repaired = store.run("embedding/contrast", recompute)

    assert calls == 1
    assert repaired.reused is False
    np.testing.assert_array_equal(repaired.value, np.arange(6, dtype=float) + 100)
    assert any(checkpoint_dir.parent.glob(f".{first.fingerprint}.invalid.*"))


def test_context_or_row_identity_change_invalidates_without_scanning_payloads(tmp_path):
    first = _store(tmp_path, identity="config-a")
    first.run(
        "unit",
        lambda: np.asarray([1.0]),
        rows={"fit": row_id_identity([1, 2, 3])},
    )
    calls = 0

    def compute():
        nonlocal calls
        calls += 1
        return np.asarray([2.0])

    changed_config = _store(tmp_path, identity="config-b")
    changed_config.run(
        "unit",
        compute,
        rows={"fit": row_id_identity([1, 2, 3])},
    )
    changed_rows = _store(tmp_path, identity="config-a")
    changed_rows.run(
        "unit",
        compute,
        rows={"fit": row_id_identity([3, 2, 1])},
    )

    assert calls == 2


def test_each_leaf_seed_is_independent_of_resume_order(tmp_path):
    def random_vector():
        return np.concatenate([np.random.random(3), torch.rand(2).numpy()])

    interrupted = _store(tmp_path / "interrupted")
    first_a = interrupted.run(
        "htr/nuisance/fold_001",
        random_vector,
        deterministic_seed=True,
    ).value
    resumed = _store(tmp_path / "interrupted")
    resumed_a = resumed.run(
        "htr/nuisance/fold_001",
        random_vector,
        deterministic_seed=True,
    ).value
    resumed_b = resumed.run(
        "htr/nuisance/fold_002",
        random_vector,
        deterministic_seed=True,
    ).value

    clean = _store(tmp_path / "clean")
    clean_a = clean.run(
        "htr/nuisance/fold_001",
        random_vector,
        deterministic_seed=True,
    ).value
    clean_b = clean.run(
        "htr/nuisance/fold_002",
        random_vector,
        deterministic_seed=True,
    ).value

    np.testing.assert_array_equal(first_a, resumed_a)
    np.testing.assert_array_equal(first_a, clean_a)
    np.testing.assert_array_equal(resumed_b, clean_b)


def test_htr_nuisance_folds_resume_after_interruption(tmp_path, monkeypatch):
    monkeypatch.setattr(multi_model_forest_stage1, "_NuisanceNet", _FakeHTRNet)
    train = pd.DataFrame(
        {
            "_oci_row_id": np.arange(9),
            "treatment": [0, 1, 0, 1, 0, 1, 0, 1, 0],
            "outcome": [1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    test = pd.DataFrame(
        {"_oci_row_id": [20, 21], "treatment": [0, 1], "outcome": [1, 0]}
    )
    first_runner = _FakeHTRRunner(interrupt_stage="nuisance", interrupt_fold=2)
    first_provider = _htr_provider(_store(tmp_path / "interrupted"), first_runner)
    with pytest.raises(RuntimeError, match="simulated HTR nuisance interruption"):
        first_provider.fit_nuisance_inner_ensemble_predict(
            train,
            test,
            outer_fold=1,
        )
    assert first_runner.nuisance_calls == [1, 2]

    resumed_runner = _FakeHTRRunner()
    resumed = _htr_provider(
        _store(tmp_path / "interrupted"),
        resumed_runner,
    ).fit_nuisance_inner_ensemble_predict(train, test, outer_fold=1)
    assert resumed_runner.nuisance_calls == [2, 3]

    clean_runner = _FakeHTRRunner()
    clean = _htr_provider(
        _store(tmp_path / "clean"),
        clean_runner,
    ).fit_nuisance_inner_ensemble_predict(train, test, outer_fold=1)
    assert clean_runner.nuisance_calls == [1, 2, 3]
    pd.testing.assert_frame_equal(
        resumed["train"]["predictions"],
        clean["train"]["predictions"],
    )
    pd.testing.assert_frame_equal(resumed["test_predictions"], clean["test_predictions"])
    assert resumed["train"]["attention"] == clean["train"]["attention"]
    assert resumed["inner_model_rows"] == clean["inner_model_rows"]


def test_htr_effect_folds_resume_after_interruption(tmp_path, monkeypatch):
    monkeypatch.setattr(multi_model_forest_stage1, "_EffectNet", _FakeHTRNet)
    train = pd.DataFrame(
        {
            "_oci_row_id": np.arange(9),
            "treatment": [0, 1, 0, 1, 0, 1, 0, 1, 0],
            "outcome": [1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    test = pd.DataFrame(
        {"_oci_row_id": [20, 21], "treatment": [0, 1], "outcome": [1, 0]}
    )
    nuisance = pd.DataFrame(
        {
            "_oci_row_id": np.arange(9),
            "e_hat": np.linspace(0.2, 0.8, 9),
            "m_hat": np.linspace(0.3, 0.7, 9),
        }
    )
    test_nuisance = pd.DataFrame(
        {"_oci_row_id": [20, 21], "e_hat": [0.4, 0.6], "m_hat": [0.6, 0.4]}
    )

    def prepare(store):
        store.run("assembly/nuisance_ensemble", lambda: {"ready": True})
        return store

    first_runner = _FakeHTRRunner(interrupt_stage="effect", interrupt_fold=2)
    with pytest.raises(RuntimeError, match="simulated HTR effect interruption"):
        _htr_provider(
            prepare(_store(tmp_path / "interrupted")),
            first_runner,
        ).fit_effect_variant_inner_ensemble_predict(
            train,
            test,
            nuisance,
            outer_fold=1,
            effect_objective="squared_r_loss",
            test_nuisance_predictions=test_nuisance,
        )
    assert first_runner.effect_calls == [1, 2]

    resumed_runner = _FakeHTRRunner()
    resumed = _htr_provider(
        prepare(_store(tmp_path / "interrupted")),
        resumed_runner,
    ).fit_effect_variant_inner_ensemble_predict(
        train,
        test,
        nuisance,
        outer_fold=1,
        effect_objective="squared_r_loss",
        test_nuisance_predictions=test_nuisance,
    )
    assert resumed_runner.effect_calls == [2, 3]

    clean_runner = _FakeHTRRunner()
    clean = _htr_provider(
        prepare(_store(tmp_path / "clean")),
        clean_runner,
    ).fit_effect_variant_inner_ensemble_predict(
        train,
        test,
        nuisance,
        outer_fold=1,
        effect_objective="squared_r_loss",
        test_nuisance_predictions=test_nuisance,
    )
    assert clean_runner.effect_calls == [1, 2, 3]
    pd.testing.assert_frame_equal(
        resumed["train"]["predictions"],
        clean["train"]["predictions"],
    )
    pd.testing.assert_frame_equal(resumed["test_predictions"], clean["test_predictions"])
    assert resumed["train"]["attention"] == clean["train"]["attention"]
    assert resumed["inner_model_rows"] == clean["inner_model_rows"]


def test_context_lock_rejects_overlapping_writer(tmp_path):
    owner = _store(tmp_path)
    contender = _store(tmp_path)

    with owner.context_lock():
        with pytest.raises(Stage1CheckpointBusyError, match="already running"):
            with contender.context_lock():
                raise AssertionError("contender unexpectedly acquired the lock")


def test_native_capture_never_reuses_checkpointed_outputs(tmp_path):
    store = _store(tmp_path)
    store.run("bow/unit", lambda: np.asarray([1.0]))
    runner = object.__new__(multi_model_forest_stage1.MultiModelForestStage1Runner)
    runner.checkpoint_store = store
    runner.bow_native_capture_sink = object()
    runner.htr_native_capture_sink = None
    runner.matched_pair_native_capture_sink = None
    runner.embedding_provider = None
    runner.htr_evidence_provider = None
    calls = 0

    def compute():
        nonlocal calls
        calls += 1
        return np.asarray([2.0])

    value = runner._checkpointed_value(
        "bow/unit",
        compute,
        train_df=pd.DataFrame({"_oci_row_id": [1]}),
        test_df=pd.DataFrame({"_oci_row_id": [2]}),
    )

    assert calls == 1
    np.testing.assert_array_equal(value, [2.0])


def test_dataset_identity_reads_only_path_size_and_mtime(tmp_path):
    dataset = tmp_path / "dataset.parquet"
    dataset.write_bytes(b"not a real parquet file")

    identity = dataset_file_identity(dataset)

    assert identity == {
        "path": str(dataset.resolve()),
        "size": dataset.stat().st_size,
        "mtime_ns": dataset.stat().st_mtime_ns,
    }


def test_htr_pair_folds_resume_after_interruption_with_equivalent_result(
    tmp_path,
    monkeypatch,
):
    train = pd.DataFrame({"_oci_row_id": np.arange(9)})
    test = pd.DataFrame({"_oci_row_id": [20, 21]})
    y = np.asarray([0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=float)
    t = np.asarray([0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=float)
    nuisance_train = np.linspace(0.2, 0.8, len(train))
    nuisance_test = np.asarray([0.35, 0.65])

    def fold_payload(**kwargs):
        heldout_pos = np.asarray(kwargs["heldout_pos"], dtype=int)
        inner_fold = int(kwargs["inner_fold"])
        return {
            "heldout_pos": heldout_pos,
            "fold_delta": heldout_pos.astype(float) + inner_fold / 10.0,
            "fold_prob": np.full(len(heldout_pos), 0.4 + inner_fold / 20.0),
            "fold_n": np.full(len(heldout_pos), inner_fold, dtype=float),
            "test_delta": np.full(len(test), inner_fold / 10.0),
            "test_prob": np.full(len(test), 0.5 + inner_fold / 20.0),
            "test_n": np.full(len(test), inner_fold, dtype=float),
            "attention_rows": [{"inner_fold": inner_fold}],
            "evidence": {"inner_fold": inner_fold},
            "prediction_frame": pd.DataFrame(),
        }

    def run(store):
        store.run("assembly/nuisance_ensemble", lambda: {"ready": True})
        return multi_model_pair_uplift.fit_htr_pair_uplift_train_test(
            runner=object(),
            train_df=train,
            test_df=test,
            texts_train=[f"train {index}" for index in range(len(train))],
            texts_test=["test a", "test b"],
            y_train=y,
            t_train=t,
            e_train=nuisance_train,
            m_train=nuisance_train,
            e_test=nuisance_test,
            m_test=nuisance_test,
            outer_fold=1,
            effect_folds=3,
            propensity_caliper=0.1,
            outcome_caliper=0.1,
            max_controls_per_candidate=2,
            nearest_fallback_controls=1,
            max_attention_pairs=4,
            checkpoint_store=store,
            checkpoint_dependencies=("assembly/nuisance_ensemble",),
        )

    interrupted_store = _store(tmp_path / "interrupted")
    first_calls: list[int] = []

    def interrupting_fold(**kwargs):
        inner_fold = int(kwargs["inner_fold"])
        first_calls.append(inner_fold)
        if inner_fold == 2:
            raise RuntimeError("simulated pair interruption")
        return fold_payload(**kwargs)

    monkeypatch.setattr(
        multi_model_pair_uplift,
        "_fit_htr_pair_uplift_fold",
        interrupting_fold,
    )
    with pytest.raises(RuntimeError, match="simulated pair interruption"):
        run(interrupted_store)
    assert first_calls == [1, 2]

    resumed_calls: list[int] = []

    def resumed_fold(**kwargs):
        resumed_calls.append(int(kwargs["inner_fold"]))
        return fold_payload(**kwargs)

    monkeypatch.setattr(
        multi_model_pair_uplift,
        "_fit_htr_pair_uplift_fold",
        resumed_fold,
    )
    resumed = run(_store(tmp_path / "interrupted"))
    assert resumed_calls == [2, 3]

    clean_calls: list[int] = []

    def clean_fold(**kwargs):
        clean_calls.append(int(kwargs["inner_fold"]))
        return fold_payload(**kwargs)

    monkeypatch.setattr(
        multi_model_pair_uplift,
        "_fit_htr_pair_uplift_fold",
        clean_fold,
    )
    clean = run(_store(tmp_path / "clean"))
    assert clean_calls == [1, 2, 3]
    np.testing.assert_array_equal(resumed.train_delta_logit, clean.train_delta_logit)
    np.testing.assert_array_equal(resumed.test_delta_logit, clean.test_delta_logit)
    np.testing.assert_array_equal(resumed.train_pred_prob, clean.train_pred_prob)
    assert resumed.evidence_rows == clean.evidence_rows
    assert resumed.attention_rows == clean.attention_rows
    assert resumed.metrics == clean.metrics


def test_bow_pair_folds_and_full_importance_resume_independently(tmp_path, monkeypatch):
    train = pd.DataFrame({"_oci_row_id": np.arange(9)})
    test = pd.DataFrame({"_oci_row_id": [20, 21]})
    y = np.asarray([0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=float)
    t = np.asarray([0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=float)
    nuisance_train = np.linspace(0.2, 0.8, len(train))
    nuisance_test = np.asarray([0.35, 0.65])

    def fold_payload(**kwargs):
        heldout_pos = np.asarray(kwargs["heldout_pos"], dtype=int)
        inner_fold = int(kwargs["inner_fold"])
        return {
            "heldout_pos": heldout_pos,
            "fold_delta": heldout_pos.astype(float) + inner_fold / 10.0,
            "fold_prob": np.full(len(heldout_pos), 0.4 + inner_fold / 20.0),
            "fold_n": np.full(len(heldout_pos), inner_fold, dtype=float),
            "test_delta": np.full(len(test), inner_fold / 10.0),
            "test_prob": np.full(len(test), 0.5 + inner_fold / 20.0),
            "test_n": np.full(len(test), inner_fold, dtype=float),
            "attention_rows": [],
            "evidence": {"inner_fold": inner_fold},
            "prediction_frame": pd.DataFrame(),
        }

    def run(store):
        store.run("assembly/nuisance_ensemble", lambda: {"ready": True})
        return multi_model_pair_uplift.fit_bow_pair_uplift_train_test(
            train_df=train,
            test_df=test,
            texts_train=[f"train {index}" for index in range(len(train))],
            texts_test=["test a", "test b"],
            y_train=y,
            t_train=t,
            e_train=nuisance_train,
            m_train=nuisance_train,
            e_test=nuisance_test,
            m_test=nuisance_test,
            vectorizer_params={},
            model_params={},
            outer_fold=1,
            view_name="test_view",
            view_index=0,
            effect_folds=3,
            propensity_caliper=0.1,
            outcome_caliper=0.1,
            max_controls_per_candidate=2,
            nearest_fallback_controls=1,
            l2_alpha=1.0,
            max_iter=10,
            top_n=5,
            checkpoint_store=store,
            checkpoint_dependencies=("assembly/nuisance_ensemble",),
            checkpoint_namespace="bow/pair_uplift/view_000",
        )

    importance_calls: list[str] = []

    def importance(**_kwargs):
        importance_calls.append("fit")
        return {
            "importance": {"view_name": "test_view", "positive": []},
            "n_matched_training_pairs": 7,
        }

    monkeypatch.setattr(
        multi_model_pair_uplift,
        "_fit_bow_pair_full_importance",
        importance,
    )
    first_calls: list[int] = []

    def interrupting_fold(**kwargs):
        inner_fold = int(kwargs["inner_fold"])
        first_calls.append(inner_fold)
        if inner_fold == 2:
            raise RuntimeError("simulated BoW pair interruption")
        return fold_payload(**kwargs)

    monkeypatch.setattr(
        multi_model_pair_uplift,
        "_fit_bow_pair_uplift_fold",
        interrupting_fold,
    )
    with pytest.raises(RuntimeError, match="simulated BoW pair interruption"):
        run(_store(tmp_path / "interrupted"))
    assert first_calls == [1, 2]
    assert importance_calls == []

    resumed_calls: list[int] = []

    def resumed_fold(**kwargs):
        resumed_calls.append(int(kwargs["inner_fold"]))
        return fold_payload(**kwargs)

    monkeypatch.setattr(
        multi_model_pair_uplift,
        "_fit_bow_pair_uplift_fold",
        resumed_fold,
    )
    resumed = run(_store(tmp_path / "interrupted"))
    assert resumed_calls == [2, 3]
    assert importance_calls == ["fit"]

    clean_calls: list[int] = []

    def clean_fold(**kwargs):
        clean_calls.append(int(kwargs["inner_fold"]))
        return fold_payload(**kwargs)

    monkeypatch.setattr(
        multi_model_pair_uplift,
        "_fit_bow_pair_uplift_fold",
        clean_fold,
    )
    clean = run(_store(tmp_path / "clean"))
    assert clean_calls == [1, 2, 3]
    assert importance_calls == ["fit", "fit"]
    np.testing.assert_array_equal(resumed.train_delta_logit, clean.train_delta_logit)
    np.testing.assert_array_equal(resumed.test_delta_logit, clean.test_delta_logit)
    assert resumed.feature_importance == clean.feature_importance
    assert resumed.metrics == clean.metrics
