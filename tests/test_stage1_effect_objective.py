"""HTR effect_objective: "auto" by outcome type, explicit mismatches raise (B10)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from oci.config import (
    AgenticAttentionVariableForestConfig,
    AppliedInferenceConfig,
    ExperimentConfig,
    ModelArchitectureConfig,
    resolve_effect_objective,
)
from oci.inference import multi_model_forest_stage1
from oci.inference.htr_trainer import HTRTrainer, effect_objective_name

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("value", "outcome_type", "expected"),
    [
        ("auto", "binary", "dina"),
        ("auto", "continuous", "pseudo_outcome_mse"),
        ("dina", "binary", "dina"),
        (" Squared_R_Loss ", "continuous", "squared_r_loss"),
        ("pseudo_outcome_mse", "continuous", "pseudo_outcome_mse"),
    ],
)
def test_effect_objective_resolution(value, outcome_type, expected):
    assert resolve_effect_objective(value, outcome_type=outcome_type) == expected


@pytest.mark.parametrize("value", ["pseudo_outcome_mse", "squared_r_loss"])
def test_binary_outcomes_reject_non_dina_objectives(value):
    with pytest.raises(ValueError, match="binary outcomes require"):
        resolve_effect_objective(value, outcome_type="binary")


def test_continuous_outcomes_reject_dina_and_unknown_objectives():
    with pytest.raises(ValueError, match="requires outcome_type='binary'"):
        resolve_effect_objective("dina", outcome_type="continuous")
    with pytest.raises(ValueError, match="must be one of"):
        resolve_effect_objective("logistic_r_loss", outcome_type="binary")


def _applied(outcome_type: str, effect_objective: str) -> AppliedInferenceConfig:
    return AppliedInferenceConfig(
        outcome_type=outcome_type,
        architecture=ModelArchitectureConfig(
            agentic_attention_variable_forest=AgenticAttentionVariableForestConfig(
                effect_objective=effect_objective
            )
        ),
    )


def test_default_is_auto_and_trainer_resolves_it_without_mutating_config():
    assert AgenticAttentionVariableForestConfig().effect_objective == "auto"
    for outcome_type, expected in (("binary", "dina"), ("continuous", "pseudo_outcome_mse")):
        config = AppliedInferenceConfig(outcome_type=outcome_type)
        trainer = HTRTrainer(config=config, device="cpu")
        assert trainer.avf_config.effect_objective == expected
        assert effect_objective_name(trainer.avf_config) == expected
        assert config.architecture.agentic_attention_variable_forest.effect_objective == "auto"


def test_explicit_mismatch_fails_at_config_parse_time():
    with pytest.raises(ValueError, match="binary outcomes require"):
        _applied("binary", "pseudo_outcome_mse")
    with pytest.raises(ValueError, match="binary outcomes require"):
        ExperimentConfig.from_dict(
            {
                "applied_inference": {
                    "outcome_type": "binary",
                    "architecture": {
                        "agentic_attention_variable_forest": {
                            "effect_objective": "squared_r_loss"
                        }
                    },
                }
            }
        )
    assert _applied("continuous", "squared_r_loss").outcome_type == "continuous"


def test_trainer_rejects_a_mismatch_introduced_after_parsing():
    config = _applied("continuous", "squared_r_loss")
    config.outcome_type = "binary"
    with pytest.raises(ValueError, match="binary outcomes require"):
        HTRTrainer(config=config, device="cpu")


def test_unresolved_auto_is_not_silently_interpreted():
    with pytest.raises(ValueError, match="must be resolved"):
        effect_objective_name(AgenticAttentionVariableForestConfig())


def test_stage1_template_uses_auto_and_parses_for_both_outcome_types():
    template = json.loads(
        (REPOSITORY_ROOT / "example_configs" / "all_evidence_stage1_model.json").read_text(
            encoding="utf-8"
        )
    )
    avf_section = template["config"]["architecture"]["agentic_attention_variable_forest"]
    assert avf_section["effect_objective"] == "auto"
    for outcome_type in ("binary", "continuous"):
        config = AppliedInferenceConfig(
            outcome_type=outcome_type,
            architecture=ModelArchitectureConfig(
                agentic_attention_variable_forest=AgenticAttentionVariableForestConfig(
                    **avf_section
                )
            ),
        )
        assert HTRTrainer(config=config, device="cpu").avf_config.effect_objective == (
            "dina" if outcome_type == "binary" else "pseudo_outcome_mse"
        )


def test_stage1_effect_variant_mismatch_raises_instead_of_running_dina():
    runner = SimpleNamespace(
        avf_config=SimpleNamespace(effect_folds=3, effect_objective="dina"),
        config=SimpleNamespace(outcome_type="binary"),
    )
    provider = object.__new__(multi_model_forest_stage1.MultiModelForestStage1HTRProvider)
    provider._runner = runner
    provider.config = runner.config
    provider.checkpoint_store = None
    provider._ensure_runner = lambda _frame: runner
    train = pd.DataFrame({"_oci_row_id": np.arange(6)})

    with pytest.raises(ValueError, match="binary outcomes require"):
        provider.fit_effect_variant_inner_ensemble_predict(
            train,
            train.iloc[:2],
            pd.DataFrame({"_oci_row_id": np.arange(6), "e_hat": 0.5, "m_hat": 0.5}),
            outer_fold=1,
            effect_objective="squared_r_loss",
        )
    assert runner.avf_config.effect_objective == "dina"
