from __future__ import annotations

import json
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

    def _dina_effect_predictions(self, model, frame, delta):
        # HTRTrainer's default hook, so tests can patch stage1_dina.neural_predictions.
        from oci.inference import stage1_dina

        return stage1_dina.neural_predictions(self, model, frame, delta)

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


def _htr_provider(store: Stage1CheckpointStore, runner: _FakeHTRRunner):
    provider = object.__new__(multi_model_forest_stage1.MultiModelForestStage1HTRProvider)
    provider._runner = runner
    provider.config = runner.config
    provider.checkpoint_store = store
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


def test_retire_makes_every_checkpoint_of_one_unit_unloadable_and_keeps_payloads(tmp_path):
    def not_recomputed():
        pytest.fail("a restorable unit was recomputed")

    under_a = _store(tmp_path, identity="config-a").run("assembly", lambda: np.asarray([1.0]))
    under_b = _store(tmp_path, identity="config-b").run("assembly", lambda: np.asarray([2.0]))
    _store(tmp_path, identity="config-a").run("assembly/part", lambda: np.asarray([3.0]))
    _store(tmp_path, identity="config-a").run("bow/fit", lambda: np.asarray([4.0]))
    unit_dir = tmp_path / "checkpoints" / "v1" / "units" / "assembly"

    _store(tmp_path, identity="config-b").retire("assembly")
    _store(tmp_path, identity="config-b").retire("never/computed")

    for result in (under_a, under_b):
        assert not (unit_dir / result.fingerprint / "complete.json").exists()
        assert (unit_dir / result.fingerprint / "payload.json").is_file()
    store = _store(tmp_path, identity="config-a")
    assert store.run("assembly/part", not_recomputed).reused is True
    assert store.run("bow/fit", not_recomputed).reused is True
    recomputed = store.run("assembly", lambda: np.asarray([5.0]))
    assert recomputed.reused is False
    assert recomputed.fingerprint == under_a.fingerprint
    np.testing.assert_array_equal(recomputed.value, [5.0])


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
    monkeypatch.setattr(multi_model_forest_stage1, "NuisanceNet", _FakeHTRNet)
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


def test_htr_effect_folds_resume_after_interruption(tmp_path, monkeypatch):
    from oci.inference import stage1_dina
    from oci.models import dina

    def predictions(runner, model, frame, delta):
        nuisance = dina.nuisances(
            np.full(len(frame), 0.5), np.full(len(frame), 0.3), np.full(len(frame), 0.6)
        )
        mu0, mu1 = dina.counterfactuals(delta, nuisance["a"], nuisance["nu"])
        return {**nuisance, "delta": delta, "tau": mu1 - mu0, "mu0": mu0, "mu1": mu1}

    monkeypatch.setattr(stage1_dina, "neural_predictions", predictions)

    monkeypatch.setattr(multi_model_forest_stage1, "EffectNet", _FakeHTRNet)
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
            effect_objective="dina",
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
        effect_objective="dina",
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
        effect_objective="dina",
    )
    assert clean_runner.effect_calls == [1, 2, 3]
    pd.testing.assert_frame_equal(
        resumed["train"]["predictions"],
        clean["train"]["predictions"],
    )
    pd.testing.assert_frame_equal(resumed["test_predictions"], clean["test_predictions"])
    assert resumed["train"]["attention"] == clean["train"]["attention"]


def test_context_lock_rejects_overlapping_writer(tmp_path):
    owner = _store(tmp_path)
    contender = _store(tmp_path)

    with owner.context_lock():
        with pytest.raises(Stage1CheckpointBusyError, match="already running"):
            with contender.context_lock():
                raise AssertionError("contender unexpectedly acquired the lock")


def test_dataset_identity_reads_only_path_size_and_mtime(tmp_path):
    dataset = tmp_path / "dataset.parquet"
    dataset.write_bytes(b"not a real parquet file")

    identity = dataset_file_identity(dataset)

    assert identity == {
        "path": str(dataset.resolve()),
        "size": dataset.stat().st_size,
        "mtime_ns": dataset.stat().st_mtime_ns,
    }


def test_matched_pair_vectorizer_units_resume_and_carry_the_lane_version(
    tmp_path, monkeypatch
):
    from oci.config import BoWViewConfig

    rng = np.random.default_rng(4)
    n = 160
    train = pd.DataFrame({"_oci_row_id": np.arange(n)})
    test = pd.DataFrame({"_oci_row_id": [500, 501]})
    t = np.tile([0.0, 1.0], n // 2)
    y = (rng.random(n) < 0.5).astype(float)
    texts = [f"word{index % 7} term{index % 3}" for index in range(n)]
    views = [
        BoWViewConfig(name="uni", ngram_range_min=1, ngram_range_max=1, min_df=1),
        BoWViewConfig(name="bi", ngram_range_min=1, ngram_range_max=2, min_df=1),
    ]
    groups = multi_model_pair_uplift.distinct_vectorizer_groups(views)
    settings = multi_model_pair_uplift.MatchedPairSettings.from_config(
        multi_model_forest_stage1.MultiModelForestConfig(
            bow_views=views, matched_pair_min_discordant_pairs=4
        ),
        outcome_type="binary",
        top_n=5,
    )
    calls: list[str] = []

    def fake_view(**kwargs):
        name = str(kwargs["group"]["name"])
        calls.append(name)
        if name == "bi" and interrupt["active"]:
            raise RuntimeError("simulated matched-pair interruption")
        return {
            "view": {"view_name": f"matched_pair__{name}", "status": "fitted"},
            "test_tau": np.full(len(test), float(len(name))),
        }

    interrupt = {"active": True}
    monkeypatch.setattr(multi_model_pair_uplift, "fit_matched_pair_view", fake_view)

    def run(store):
        def checkpoint(unit, compute, *, parameters, validator):
            return store.run(unit, compute, parameters=parameters, validator=validator).value

        return multi_model_pair_uplift.run_matched_pair_lane(
            train_df=train,
            test_df=test,
            texts_train=texts,
            texts_test=["word1", "term2"],
            y_train=y,
            t_train=t,
            propensity=np.full(n, 0.5),
            prognostic=np.full(n, 0.5),
            score_sources={},
            vectorizer_groups=groups,
            settings=settings,
            outer_fold=1,
            seed=11,
            checkpoint=checkpoint,
        )

    with pytest.raises(RuntimeError, match="simulated matched-pair interruption"):
        run(_store(tmp_path))
    assert calls == ["uni", "bi"]

    interrupt["active"] = False
    resumed = run(_store(tmp_path))
    assert calls == ["uni", "bi", "bi"]
    assert [view["view_name"] for view in resumed.importance["views"]] == [
        "matched_pair__uni",
        "matched_pair__bi",
    ]
    reused = run(_store(tmp_path))
    assert calls == ["uni", "bi", "bi"]
    assert reused.importance == resumed.importance

    markers = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (tmp_path / "checkpoints" / "v1" / "units" / "bow" / "matched_pair").rglob(
            "complete.json"
        )
    ]
    assert len(markers) == 2
    for marker in markers:
        assert (
            marker["parameters"]["checkpoint_version"]
            == multi_model_pair_uplift.MATCHED_PAIR_CHECKPOINT_VERSION
            == "matched_pair_conditional_logit_checkpoint_v1"
        )
