"""Standalone joint HTR: losses, cross-fitting, resume, and oracle separation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from oci.inference import standalone_joint_htr as joint_htr

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
    "training.dataloader_workers=0",
    "joint.outer_folds=2",
    "joint.epochs=2",
    "joint.effect_warmup_epochs=1",
    "joint.validation_fraction=0.2",
]


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
    return joint_htr.build_configs(
        outcome_type="binary", htr_model="hash", overrides=[*TINY_HTR, *extra]
    )


def test_effect_loss_is_pseudo_outcome_mse_with_detached_nuisances():
    t_logit = torch.tensor([0.3, -1.0, 2.0, 0.0], requires_grad=True)
    m_output = torch.tensor([-0.2, 0.5, 1.0, 0.1], requires_grad=True)
    tau = torch.tensor([0.1, -0.2, 0.3, 0.0], requires_grad=True)
    t = torch.tensor([1.0, 0.0, 1.0, 0.0])
    y = torch.tensor([1.0, 1.0, 0.0, 0.0])
    parts = joint_htr.joint_losses(
        t_logit, m_output, tau, t, y, outcome_type="binary", e_clip=0.2, label_smoothing=0.0
    )
    e = torch.sigmoid(t_logit).detach().clamp(0.2, 0.8)
    pseudo = (y - torch.sigmoid(m_output).detach()) / (t - e)
    assert torch.allclose(parts["effect"], torch.mean((tau.detach() - pseudo) ** 2))
    assert parts["e_clipped_fraction"].item() == pytest.approx(0.25)  # sigmoid(2.0) > 0.8
    parts["effect"].backward()
    assert t_logit.grad is None and m_output.grad is None
    assert tau.grad is not None and torch.any(tau.grad != 0)


def test_validation_free_runs_require_last_epoch_and_no_calibration():
    with pytest.raises(ValueError, match="validation_fraction=0"):
        joint_htr.JointHTRConfig(validation_fraction=0.0)
    joint_htr.JointHTRConfig(
        validation_fraction=0.0, selection="last_epoch", nuisance_calibration="none"
    )
    with pytest.raises(ValueError, match="effect_warmup_epochs"):
        joint_htr.JointHTRConfig(epochs=3, effect_warmup_epochs=3)


def test_unknown_settings_are_rejected():
    with pytest.raises(ValueError, match="unknown joint settings"):
        _configs("joint.not_a_setting=1")
    with pytest.raises(ValueError, match="architecture.<field>"):
        _configs("stage2.model=x")


def test_run_cross_fits_every_row_without_reading_oracle_columns(tmp_path, monkeypatch):
    dataset = _cohort(tmp_path)
    out = tmp_path / "run"
    config, joint = _configs()
    read_columns = []
    original_read = pd.read_parquet

    def recording_read(path, *args, **kwargs):
        if Path(path).resolve() == dataset.resolve():
            read_columns.append(kwargs.get("columns"))
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", recording_read)
    summary = joint_htr.run_standalone_joint_htr(
        dataset_path=dataset, output_dir=out, config=config, joint=joint, devices=["cpu"]
    )
    assert read_columns and all(columns is not None for columns in read_columns)
    assert not any(
        joint_htr.is_oracle_column(column)
        for columns in read_columns
        for column in (columns or [])
    )

    predictions = pd.read_parquet(out / "predictions.parquet")
    assert sorted(predictions["_oci_row_id"]) == list(range(48))
    assert set(predictions["outer_fold"]) == {1, 2}
    for column in ("propensity", "m_hat", "estimated_cate", "mu0", "mu1", "aipw_score"):
        assert np.isfinite(predictions[column]).all(), column
    assert np.allclose(predictions["mu1"] - predictions["mu0"], predictions["estimated_cate"])
    assert summary["rows"] == 48
    assert set(summary["per_fold"]) == {"1", "2"}
    # Epoch selection only considers epochs after the warmup.
    assert {record["selected_epoch"] for record in summary["per_fold"].values()} == {2}
    history = json.loads((out / "folds" / "fold_01" / "history.json").read_text())
    assert [record["effect_loss_active"] for record in history] == [False, True]
    assert "validation" in history[0]

    report = joint_htr.evaluate_against_oracle(out)
    assert report["ite_column"] == "true_ite"
    assert report["cate"]["rows"] == 48
    assert (out / "evaluation" / "oracle_metrics.json").exists()


def test_rerun_reuses_finished_folds_and_rejects_changed_settings(tmp_path, monkeypatch):
    dataset = _cohort(tmp_path)
    out = tmp_path / "run"
    config, joint = _configs()
    joint_htr.run_standalone_joint_htr(
        dataset_path=dataset, output_dir=out, config=config, joint=joint, devices=["cpu"]
    )

    def fail(**kwargs):
        raise AssertionError("a finished fold was refit")

    monkeypatch.setattr(joint_htr, "run_fold", fail)
    joint_htr.run_standalone_joint_htr(
        dataset_path=dataset, output_dir=out, config=config, joint=joint, devices=["cpu"]
    )
    changed_config, changed_joint = _configs("joint.effect_loss_weight=2.0")
    with pytest.raises(RuntimeError, match="different configuration"):
        joint_htr.run_standalone_joint_htr(
            dataset_path=dataset,
            output_dir=out,
            config=changed_config,
            joint=changed_joint,
            devices=["cpu"],
        )


def test_evaluation_refuses_changed_predictions(tmp_path):
    dataset = _cohort(tmp_path)
    out = tmp_path / "run"
    config, joint = _configs()
    joint_htr.run_standalone_joint_htr(
        dataset_path=dataset, output_dir=out, config=config, joint=joint, devices=["cpu"]
    )
    predictions = pd.read_parquet(out / "predictions.parquet")
    predictions.loc[0, "estimated_cate"] += 1.0
    predictions.to_parquet(out / "predictions.parquet", index=False)
    with pytest.raises(RuntimeError, match="does not match"):
        joint_htr.evaluate_against_oracle(out)


def test_split_provenance_file_folds_are_reused(tmp_path):
    dataset = _cohort(tmp_path)
    config, joint = _configs()
    frame = joint_htr.load_training_frame(dataset, config)
    folds, _ = joint_htr.outer_splits(frame, config, joint)
    splits = tmp_path / "split_provenance.jsonl"
    splits.write_text(
        "".join(
            json.dumps(
                {"outer_fold": fold, "fit_row_ids": fit.tolist(), "heldout_row_ids": test.tolist()}
            )
            + "\n"
            for fold, fit, test in folds
        )
    )
    reused, metadata = joint_htr.outer_splits(frame, config, joint, splits)
    assert metadata["method"] == "split_provenance_file"
    for (_, fit, test), (_, fit2, test2) in zip(folds, reused, strict=True):
        assert np.array_equal(np.sort(fit), np.sort(fit2))
        assert np.array_equal(np.sort(test), np.sort(test2))
    splits.write_text(json.dumps({"outer_fold": 1, "fit_row_ids": [0], "heldout_row_ids": [1]}))
    with pytest.raises(ValueError, match="cover every row"):
        joint_htr.outer_splits(frame, config, joint, splits)


def test_folds_run_in_parallel_device_processes(tmp_path):
    dataset = _cohort(tmp_path)
    out = tmp_path / "run"
    config, joint = _configs()
    summary = joint_htr.run_standalone_joint_htr(
        dataset_path=dataset, output_dir=out, config=config, joint=joint, devices=["cpu", "cpu"]
    )
    assert summary["rows"] == 48
    devices = {
        json.loads((out / "folds" / f"fold_0{fold}" / "fold.json").read_text())["device"]
        for fold in (1, 2)
    }
    assert devices == {"cpu"}
    assert sorted(path.name for path in (out / "logs").iterdir()) == ["worker_cpu.log"]
