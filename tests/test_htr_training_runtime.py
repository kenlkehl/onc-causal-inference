"""HTR prefetching, reproducibility, and epoch-only metric transfers."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from joblib import Parallel, delayed

from oci.config import AppliedInferenceConfig, TrainingConfig, create_default_config
from oci.inference.htr_trainer import (
    EffectNet,
    HTRTrainer,
    NuisanceNet,
    torch_pseudo_outcome_mse_loss_vector,
)
from oci.inference.multi_model_forest_stage1 import MultiModelForestStage1Runner
from oci.models.hierarchical_transformer_extractor import HierarchicalTransformerBatchPreprocessor


def _frame():
    return pd.DataFrame(
        {
            "clinical_text": ["1 2", "3 4 5", "6", "7 8", "9 10 11"],
            "treatment_indicator": [0, 1, 0, 1, 0],
            "outcome_indicator": [1, 0, 1, 1, 0],
        }
    )


class _Tokenizer:
    pad_token_id = 0
    padding_side = "right"

    def __call__(self, text, **kwargs):
        ids = [int(word) for word in text.split()]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class _Preprocessor(HierarchicalTransformerBatchPreprocessor):
    def __call__(self, texts):
        batch = super().__call__(texts)
        batch["worker_pid"] = os.getpid()
        return batch


class _PreparedExtractor:
    def make_batch_preprocessor(self):
        return _Preprocessor(
            tokenizer=_Tokenizer(),
            chunk_size_words=4,
            chunk_overlap_words=0,
            max_chunks=4,
            max_chunk_length=8,
            tokenize_for_transformers=True,
            chunk_cache_max_entries=100,
            tokenization_cache_max_entries=100,
        )


class _PreparedModel:
    extractor = _PreparedExtractor()


def _collect_prefetched_epochs(workers):
    """Also called inside loky: exercises the production nesting of spawn."""
    torch.manual_seed(918)
    config = AppliedInferenceConfig(
        training=TrainingConfig(batch_size=2, dataloader_workers=workers)
    )
    trainer = HTRTrainer(config=config, device="cpu", num_workers=3)
    frame = _frame()
    loader = trainer._make_text_loader(
        _PreparedModel(),
        frame,
        np.arange(len(frame)),
        fields={"y": frame.outcome_indicator.to_numpy()},
        shuffle=True,
    )
    rng_before = torch.get_rng_state().clone()
    epochs, pids = [], []
    try:
        for _ in range(2):
            records, worker_pids = [], set()
            for batch in loader:
                prepared = batch["model_input"]
                worker_pids.add(prepared["worker_pid"])
                assert prepared["chunk_input_ids"].device.type == "cpu"
                for index, position in enumerate(batch["position"].tolist()):
                    ids = prepared["chunk_input_ids"][index]
                    mask = prepared["chunk_attention_mask"][index].bool()
                    records.append((position, ids[mask].tolist(), batch["y"][index].item()))
            epochs.append(records)
            pids.append(sorted(worker_pids))
        return epochs, pids, os.getpid(), torch.equal(rng_before, torch.get_rng_state())
    finally:
        # Explicit shutdown makes the test fail here if nested workers hang,
        # and ensures no test workers outlive their enclosing loky job.
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()


def test_spawn_prefetch_inside_loky_preserves_rows_rng_and_persistent_workers():
    serial = _collect_prefetched_epochs(0)
    with Parallel(n_jobs=2, backend="loky") as pool:
        prefetched = pool(delayed(_collect_prefetched_epochs)(2) for _ in range(2))
    for epochs, pids, parent_pid, rng_unchanged in prefetched:
        assert epochs == serial[0]
        assert rng_unchanged and serial[3]
        assert pids[0] == pids[1]
        assert len(pids[0]) == 2
        assert parent_pid not in pids[0]
        for records in epochs:
            assert sorted(row[0] for row in records) == list(range(5))
            for position, tokens, outcome in records:
                assert tokens == [int(x) for x in _frame().clinical_text[position].split()]
                assert outcome == _frame().outcome_indicator[position]


@pytest.mark.parametrize("budget,expected", [(None, 2), (0, 0), (1, 0), (2, 1), (3, 2), (30, 2)])
def test_loader_worker_budget(budget, expected):
    trainer = HTRTrainer(config=AppliedInferenceConfig(), device="cpu", num_workers=budget)
    assert trainer._data_loader_workers(total_folds=5) == expected


def test_stage1_provider_receives_context_cpu_budget(tmp_path):
    runner = MultiModelForestStage1Runner(
        dataset=_frame(),
        config=AppliedInferenceConfig(),
        output_path=tmp_path / "predictions.parquet",
        device="cpu",
        num_workers=2,
    )
    trainer = runner._htr_provider()._ensure_runner(_frame())
    assert trainer.num_workers == 2
    assert trainer._data_loader_workers() == 1


def test_loader_synchronous_opt_out_and_custom_prefetch():
    config = AppliedInferenceConfig(
        training=TrainingConfig(dataloader_workers=0, dataloader_prefetch_factor=3)
    )
    trainer = HTRTrainer(config=config, device="cpu", num_workers=8)
    loader = trainer._make_text_loader(_PreparedModel(), _frame(), np.arange(5))
    assert loader.num_workers == 0
    assert loader.prefetch_factor is None
    assert loader.multiprocessing_context is None
    assert not loader.persistent_workers
    config.training.dataloader_workers = 2
    loader = trainer._make_text_loader(_PreparedModel(), _frame(), np.arange(5))
    assert loader.prefetch_factor == 3
    assert loader.multiprocessing_context.get_start_method() == "spawn"


@pytest.mark.parametrize(
    "settings",
    [
        {"dataloader_workers": -1},
        {"dataloader_workers": True},
        {"dataloader_workers": 1.5},
        {"dataloader_prefetch_factor": 0},
        {"dataloader_prefetch_factor": False},
        {"dataloader_prefetch_factor": "2"},
    ],
)
def test_invalid_loader_settings_fail_at_configuration_time(settings):
    with pytest.raises(ValueError, match="training.dataloader_"):
        TrainingConfig(**settings)


def test_generated_and_shipped_htr_defaults_use_batches_of_8(tmp_path):
    config = TrainingConfig()
    assert config.batch_size == config.effect_batch_size == 8
    generated = tmp_path / "config.json"
    create_default_config(str(generated))
    for training in (
        json.loads(generated.read_text())["applied_inference"]["training"],
        json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "example_configs"
                / "all_evidence_stage1_model.json"
            ).read_text()
        )["config"]["training"],
    ):
        assert training["batch_size"] == training["effect_batch_size"] == 8
        assert training["dataloader_workers"] == 2


def test_pseudo_outcome_mask_preserves_invalid_row_targets_and_gradients():
    effect = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    losses, valid = torch_pseudo_outcome_mse_loss_vector(
        effect,
        torch.tensor([1.0, float("nan"), 2.0, 3.0]),
        torch.tensor([0.5, 0.5, 0.0, float("inf")]),
    )
    assert valid.tolist() == [True, False, False, False]
    torch.testing.assert_close(losses, torch.tensor([1.0, 4.0, 9.0, 16.0]))
    losses.sum().backward()
    torch.testing.assert_close(effect.grad, torch.tensor([-2.0, 4.0, 6.0, 8.0]))


class _TinyExtractor(torch.nn.Module):
    output_dim = 2

    def fit_tokenizer(self, texts):
        pass

    def forward(self, texts):
        return torch.tensor(
            [[float(text.split()[0]) / 10, len(text.split()) / 3] for text in texts]
        )


@pytest.mark.parametrize(
    "stage,outcome_type",
    [
        ("nuisance", "binary"),
        ("nuisance", "continuous"),
        ("effect", "continuous"),
        ("effect", "binary"),
    ],
)
def test_loss_metrics_transfer_only_after_complete_epochs(stage, outcome_type, monkeypatch, caplog):
    torch.manual_seed(11)
    config = AppliedInferenceConfig(
        outcome_type=outcome_type,
        training=TrainingConfig(
            batch_size=2,
            effect_batch_size=2,
            dataloader_workers=0,
            learning_rate=0,
            gradient_clip_norm=0,
        ),
    )
    trainer = HTRTrainer(config=config, device="cpu", num_workers=1)
    trainer.avf_config.nuisance_epochs = trainer.avf_config.effect_epochs = 2
    trainer.avf_config.nuisance_label_smoothing = 0
    frame = _frame()
    extractor = _TinyExtractor()
    model = (
        NuisanceNet(extractor, 3, outcome_type, head_dropout=0)
        if stage == "nuisance"
        else EffectNet(extractor, 3, head_dropout=0)
    )
    original_loader = trainer._make_text_loader

    def ordered_loader(*args, **kwargs):
        kwargs["shuffle"] = False
        return original_loader(*args, **kwargs)

    monkeypatch.setattr(trainer, "_make_text_loader", ordered_loader)
    if stage == "effect" and outcome_type == "binary":
        monkeypatch.setattr(
            "oci.inference.stage1_dina.prepare_neural_nuisance",
            lambda *a, **k: {"dina_a": np.full(5, 0.5), "dina_nu": np.zeros(5)},
        )

    # Calculate the existing mean-of-batch-means, including the short final
    # batch. A zero learning rate keeps the expected loss fixed across epochs.
    expected = []
    with torch.no_grad():
        for start in range(0, 5, 2):
            part = frame.iloc[start : start + 2]
            y = torch.tensor(part.outcome_indicator.to_numpy(), dtype=torch.float32)
            t = torch.tensor(part.treatment_indicator.to_numpy(), dtype=torch.float32)
            output = model(part.clinical_text.tolist())
            if stage == "nuisance":
                prop = torch.nn.functional.binary_cross_entropy_with_logits(output[0], t)
                outcome = (
                    torch.nn.functional.binary_cross_entropy_with_logits(output[1], y)
                    if outcome_type == "binary"
                    else torch.nn.functional.mse_loss(output[1], y)
                )
                expected.append([float(prop + outcome), float(outcome), float(prop)])
            else:
                loss = (
                    torch.nn.functional.binary_cross_entropy_with_logits((t - 0.5) * output, y)
                    if outcome_type == "binary"
                    else ((output - (y - 0.5) / (t - 0.5)) ** 2).mean()
                )
                expected.append([float(loss)])
    expected = np.mean(expected, axis=0)
    copies, steps = [], []
    original_step = trainer._clip_and_step

    def step(*args):
        original_step(*args)
        steps.append(1)

    monkeypatch.setattr(trainer, "_clip_and_step", step)
    original_cpu = torch.Tensor.cpu

    def cpu(tensor, *args, **kwargs):
        copies.append((len(steps), tensor.requires_grad, tuple(tensor.shape)))
        return original_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", cpu)
    with caplog.at_level(logging.INFO, logger="oci.inference.htr_trainer"):
        if stage == "nuisance":
            trainer._train_nuisance_model(model, frame, np.arange(5), 1, 1, 2)
        else:
            y, t = frame.outcome_indicator.to_numpy(), frame.treatment_indicator.to_numpy()
            trainer._train_effect_model(
                model,
                frame,
                np.arange(5),
                y,
                t,
                np.full(5, 0.5),
                np.full(5, 0.5),
                y - 0.5,
                t - 0.5,
                1,
                1,
                2,
            )
    assert copies == [
        (3, False, (3,) if stage == "nuisance" else ()),
        (6, False, (3,) if stage == "nuisance" else ()),
    ]
    epochs = [r for r in caplog.records if "complete:" in r.msg]
    assert len(epochs) == 2
    for record in epochs:
        values = record.args[5:8] if stage == "nuisance" else record.args[6:7]
        np.testing.assert_allclose(values, expected, rtol=1e-6)
    assert all("loss=" not in r.getMessage() for r in caplog.records if "batch %s/%s" in r.msg)
