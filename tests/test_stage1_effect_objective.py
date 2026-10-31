"""HTR effect_objective: "auto" by outcome type, explicit mismatches raise (B10)."""

from __future__ import annotations



from oci.config import (
    AgenticAttentionVariableForestConfig,
    AppliedInferenceConfig,
)
from oci.inference.htr_trainer import HTRTrainer, effect_objective_name


def test_default_is_auto_and_trainer_resolves_it_without_mutating_config():
    assert AgenticAttentionVariableForestConfig().effect_objective == "auto"
    for outcome_type, expected in (("binary", "dina"), ("continuous", "pseudo_outcome_mse")):
        config = AppliedInferenceConfig(outcome_type=outcome_type)
        trainer = HTRTrainer(config=config, device="cpu")
        assert trainer.avf_config.effect_objective == expected
        assert effect_objective_name(trainer.avf_config) == expected
        assert config.architecture.agentic_attention_variable_forest.effect_objective == "auto"
