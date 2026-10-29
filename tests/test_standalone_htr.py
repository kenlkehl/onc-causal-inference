"""Standalone HTR: two-stage and joint methods, cross-fitting, resume, and oracle separation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from oci.inference import stage1_dina, standalone_htr
from oci.inference.htr_trainer import HTRTrainer

TINY_HTR = [
    "architecture.htr_sentence_pooling=auto",
    "architecture.htr_require_live_unfrozen_encoder=false",
    "architecture.htr_num_layers=1",
    "architecture.htr_num_heads=2",
    "architecture.htr_transformer_dim=16",
    "architecture.htr_projection_dim=8",
    "architecture.htr_hash_embedding_dim=16",
    "architecture.htr_chunk_size_words=8",
    "architecture.htr_chunk_overlap_words=2",
    "architecture.htr_max_chunks=4",
    "architecture.htr_prediction_head_hidden_dim=8",
    "training.batch_size=8",
    "training.effect_batch_size=8",
    "training.dataloader_workers=0",
    "run.outer_folds=2",
    "stage1_htr.nuisance_folds=2",
    "stage1_htr.effect_folds=2",
    "stage1_htr.nuisance_epochs=1",
    "stage1_htr.effect_epochs=1",
    "joint.epochs=2",
    "joint.effect_warmup_epochs=1",
    "joint.validation_fraction=0.2",
]
PREDICTION_COLUMNS = ("propensity", "m_hat", "estimated_cate", "mu0", "mu1", "aipw_score")


def _cohort(tmp_path, n=48):
    rng = np.random.default_rng(0)
    words = ["fatigue", "cough", "stage four", "ecog one", "nausea", "pain", "egfr", "alk"]
    frame = pd.DataFrame(
        {
            "clinical_text": [" ".join(rng.choice(words, size=12)) for _ in range(n)],
            "treatment_indicator": np.tile([0, 1], n // 2),
            "outcome_indicator": np.tile([0, 0, 1, 1], n // 4),
            "true_ite": rng.normal(0.1, 0.05, size=n),
            "true_treatment_prob": np.full(n, 0.5),
            "true_y0": np.full(n, 0.4),
            "true_y1": np.full(n, 0.5),
        }
    )
    path = tmp_path / "dataset.parquet"
    frame.to_parquet(path, index=False)
    return path


def _configs(*extra):
    return standalone_htr.build_configs(
        outcome_type="binary", htr_model="hash", overrides=[*TINY_HTR, *extra]
    )


def _run(dataset, out, *extra, devices=("cpu",)):
    config, settings, joint = _configs(*extra)
    return standalone_htr.run_standalone_htr(
        dataset_path=dataset,
        output_dir=out,
        config=config,
        settings=settings,
        joint=joint,
        devices=list(devices),
    )


def _record_dataset_reads(monkeypatch, dataset):
    read_columns = []
    original_read = pd.read_parquet

    def recording_read(path, *args, **kwargs):
        if Path(path).resolve() == dataset.resolve():
            read_columns.append(kwargs.get("columns"))
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", recording_read)
    return read_columns


def _assert_no_oracle_reads(read_columns):
    assert read_columns and all(columns is not None for columns in read_columns)
    assert not any(
        standalone_htr.is_oracle_column(column) for columns in read_columns for column in columns
    )


def _assert_cross_fitted_predictions(out, rows=48):
    predictions = pd.read_parquet(out / "predictions.parquet")
    assert sorted(predictions["_oci_row_id"]) == list(range(rows))
    assert set(predictions["outer_fold"]) == {1, 2}
    for column in PREDICTION_COLUMNS:
        assert np.isfinite(predictions[column]).all(), column
    assert np.allclose(predictions["mu1"] - predictions["mu0"], predictions["estimated_cate"])
    return predictions


def test_effect_loss_is_pseudo_outcome_mse_with_detached_nuisances():
    t_logit = torch.tensor([0.3, -1.0, 2.0, 0.0], requires_grad=True)
    m_output = torch.tensor([-0.2, 0.5, 1.0, 0.1], requires_grad=True)
    tau = torch.tensor([0.1, -0.2, 0.3, 0.0], requires_grad=True)
    t = torch.tensor([1.0, 0.0, 1.0, 0.0])
    y = torch.tensor([1.0, 1.0, 0.0, 0.0])
    parts = standalone_htr.joint_losses(
        t_logit, m_output, tau, t, y, outcome_type="binary", e_clip=0.2, label_smoothing=0.0
    )
    e = torch.sigmoid(t_logit).detach().clamp(0.2, 0.8)
    pseudo = (y - torch.sigmoid(m_output).detach()) / (t - e)
    assert torch.allclose(parts["effect"], torch.mean((tau.detach() - pseudo) ** 2))
    assert parts["e_clipped_fraction"].item() == pytest.approx(0.25)  # sigmoid(2.0) > 0.8
    parts["effect"].backward()
    assert t_logit.grad is None and m_output.grad is None
    assert tau.grad is not None and torch.any(tau.grad != 0)


def _checkpoint_units(out, fold=1):
    progress = json.loads(
        (out / "folds" / f"fold_{fold:02d}" / "stage1_checkpoints" / "checkpoint_progress.json")
        .read_text()
    )
    return set(progress["units"])


def _forbid(monkeypatch, owner, *names):
    def forbidden(*args, **kwargs):
        raise AssertionError("called a function this path must not use")

    for name in names:
        monkeypatch.setattr(owner, name, forbidden)


def test_two_stage_dina_uses_htr_arm_nuisances_without_reading_oracle_columns(
    tmp_path, monkeypatch
):
    dataset = _cohort(tmp_path)
    out = tmp_path / "run"
    read_columns = _record_dataset_reads(monkeypatch, dataset)
    _forbid(monkeypatch, stage1_dina, "prepare_neural_nuisance", "neural_predictions")
    summary = _run(dataset, out)
    _assert_no_oracle_reads(read_columns)
    _assert_cross_fitted_predictions(out)
    assert summary["method"] == "two_stage"
    assert {record["effect_objective"] for record in summary["per_fold"].values()} == {"dina"}
    record = json.loads((out / "folds" / "fold_01" / "fold.json").read_text())
    assert record["inner_folds"] == {"nuisance": 2, "effect": 2}
    assert record["dina_nuisances"] == "htr"
    assert {
        "htr/arm_nuisance/fold_001",
        "htr/arm_nuisance/fold_002",
        "assembly/nuisance_ensemble",
        "htr/effect/dina/fold_001",
        "htr/effect/dina/fold_002",
    } <= _checkpoint_units(out)
    assert not any(unit.startswith("htr/nuisance/") for unit in _checkpoint_units(out))
    inner = pd.read_parquet(out / "folds" / "fold_01" / "inner_oof.parquet")
    for column in ("e_hat", "mu0_hat", "mu1_hat", "m_hat", "tau_hat_r_stage"):
        assert np.isfinite(inner[column]).all(), column
    assert np.allclose(
        inner["m_hat"], inner["e_hat"] * inner["mu1_hat"] + (1 - inner["e_hat"]) * inner["mu0_hat"]
    )
    run = json.loads((out / "run.json").read_text())
    assert run["identity"]["config"]["run"]["dina_nuisances"] == "htr"
    assert run["identity"]["config"]["run"]["dina_nuisance_clip"] == 0.01
    report = standalone_htr.evaluate_against_oracle(out)
    assert report["cate"]["rows"] == 48


def test_two_stage_fold_resumes_from_inner_network_checkpoints(tmp_path, monkeypatch):
    dataset = _cohort(tmp_path)
    out = tmp_path / "run"
    _run(dataset, out)
    first = pd.read_parquet(out / "predictions.parquet")
    # Drop the finished-fold markers but keep the Stage 1 checkpoints.
    for fold_dir in (out / "folds").iterdir():
        (fold_dir / "fold.json").unlink()
    _forbid(monkeypatch, standalone_htr, "build_arm_nuisance_net")
    _forbid(monkeypatch, HTRTrainer, "_train_nuisance_model", "_train_effect_model")
    _run(dataset, out)
    resumed = pd.read_parquet(out / "predictions.parquet")
    pd.testing.assert_frame_equal(first, resumed)
    record = json.loads((out / "folds" / "fold_01" / "fold.json").read_text())
    assert {"htr/arm_nuisance/fold_002", "htr/effect/dina/fold_002"} <= set(
        record["reused_checkpoint_units"]
    )


def test_joint_cross_fits_every_row_without_reading_oracle_columns(tmp_path, monkeypatch):
    dataset = _cohort(tmp_path)
    out = tmp_path / "run"
    read_columns = _record_dataset_reads(monkeypatch, dataset)
    summary = _run(dataset, out, "run.method=joint")
    _assert_no_oracle_reads(read_columns)
    predictions = _assert_cross_fitted_predictions(out)
    assert {"propensity_raw", "m_hat_raw"} <= set(predictions)
    assert summary["method"] == "joint"
    assert set(summary["per_fold"]) == {"1", "2"}
    # Epoch selection only considers epochs after the warmup.
    assert {record["selected_epoch"] for record in summary["per_fold"].values()} == {2}
    history = json.loads((out / "folds" / "fold_01" / "history.json").read_text())
    assert [record["effect_loss_active"] for record in history] == [False, True]
    assert "validation" in history[0]
    report = standalone_htr.evaluate_against_oracle(out)
    assert report["ite_column"] == "true_ite"
    assert report["cate"]["rows"] == 48
    assert (out / "evaluation" / "oracle_metrics.json").exists()


def test_evaluation_refuses_changed_predictions(tmp_path):
    dataset = _cohort(tmp_path)
    out = tmp_path / "run"
    _run(dataset, out, "run.method=joint")
    predictions = pd.read_parquet(out / "predictions.parquet")
    predictions.loc[0, "estimated_cate"] += 1.0
    predictions.to_parquet(out / "predictions.parquet", index=False)
    with pytest.raises(RuntimeError, match="does not match"):
        standalone_htr.evaluate_against_oracle(out)
