"""The standalone AVF runner reuses the Stage 1 HTR training core."""

from __future__ import annotations

import threading

import numpy as np
import pandas as pd
import pytest
import torch

from oci.config import AppliedInferenceConfig
from oci.inference.agentic_attention_variable_forest import AgenticAttentionVariableForestRunner
from oci.inference.htr_trainer import HTRTrainer

SHARED_METHODS = (
    "_effect_epochs",
    "_create_extractor",
    "_assert_htr_sentence_encoder_training_state",
    "_assert_htr_sentence_encoder_optimizer_coverage",
    "_make_text_loader",
    "_train_nuisance_model",
    "_train_effect_model",
    "_clip_and_step",
    "_predict_nuisance_model",
    "_predict_effect_model",
    "_attention_evidence",
    "_cleanup_model",
    "_cuda_memory_summary",
)


class _NoExtractorModel(torch.nn.Module):
    extractor = None


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "clinical_text": ["alpha beta", "gamma delta", "epsilon zeta"],
            "treatment_indicator": [0, 1, 0],
            "outcome_indicator": [1, 0, 1],
        }
    )


def _runner(tmp_path, *, num_workers: int, outcome_type: str = "continuous"):
    return AgenticAttentionVariableForestRunner(
        dataset=_frame(),
        config=AppliedInferenceConfig(outcome_type=outcome_type),
        output_path=tmp_path / "predictions.parquet",
        device=torch.device("cpu"),
        num_workers=num_workers,
        proposal_agent=object(),
        extraction_provider=object(),
    )


def test_avf_runner_inherits_the_htr_training_core_without_redefining_it():
    assert issubclass(AgenticAttentionVariableForestRunner, HTRTrainer)
    for name in SHARED_METHODS:
        assert name not in vars(AgenticAttentionVariableForestRunner), name
        assert getattr(AgenticAttentionVariableForestRunner, name) is getattr(HTRTrainer, name)
    assert "_data_loader_workers" in vars(AgenticAttentionVariableForestRunner)


@pytest.mark.parametrize("outcome_type", ["binary", "continuous"])
def test_avf_runner_initializes_shared_trainer_state(tmp_path, outcome_type):
    runner = _runner(tmp_path, num_workers=1, outcome_type=outcome_type)
    trainer = HTRTrainer(config=runner.config, device="cpu")

    assert "device" not in vars(runner)
    assert runner.device == trainer.device == torch.device("cpu")
    assert runner.avf_config == trainer.avf_config
    expected = "dina" if outcome_type == "binary" else "pseudo_outcome_mse"
    assert runner.avf_config.effect_objective == expected


def test_avf_runner_keeps_thread_local_device_for_inherited_methods(tmp_path):
    runner = _runner(tmp_path, num_workers=0)
    with runner._using_device("meta"):
        assert runner.device == torch.device("meta")
    assert runner.device == torch.device("cpu")


def test_loader_worker_policy_is_stage1_in_process_and_avf_configurable(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv("OCI_AVF_DATALOADER_WORKERS", raising=False)
    positions = np.arange(3)
    stage1 = HTRTrainer(config=AppliedInferenceConfig(), device="cpu")
    stage1_loader = stage1._make_text_loader(
        _NoExtractorModel(), _frame(), positions, total_folds=2
    )
    assert stage1_loader.num_workers == 0
    assert stage1_loader.persistent_workers is False

    runner = _runner(tmp_path, num_workers=2)
    assert threading.current_thread() is threading.main_thread()
    avf_loader = runner._make_text_loader(_NoExtractorModel(), runner.dataset, positions)
    assert avf_loader.num_workers == 2
    assert avf_loader.persistent_workers is True
    assert avf_loader.prefetch_factor == 2

    calls = []
    original = runner._data_loader_workers
    monkeypatch.setattr(
        runner,
        "_data_loader_workers",
        lambda total_folds=None: calls.append(total_folds) or original(total_folds),
    )
    runner._make_text_loader(_NoExtractorModel(), runner.dataset, positions, total_folds=5)
    assert calls == [5]
