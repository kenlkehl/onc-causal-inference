"""Bounded, opt-in search over clinically plausible measurement definitions."""

from dataclasses import asdict, dataclass
import math
from collections.abc import Mapping


@dataclass(frozen=True)
class EstimandOntologyConfig:
    enabled: bool = False
    max_features_per_role: int = 8
    max_alternatives_per_feature: int = 2
    minimum_nonmissing_fraction: float = 0.2
    minimum_relative_gain: float = 0.005
    paired_se_multiplier: float = 1.0
    minimum_winning_fold_fraction: float = 0.6
    maximum_propensity_loss_increase: float = 0.01
    maximum_overlap_fraction_drop: float = 0.05
    ridge_penalty: float = 10.0
    nuisance_crossfit_folds: int = 3
    minimum_training_rows: int = 20
    minimum_overlap_rows: int = 10
    max_prompt_chars: int = 40000

    def validate(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("estimand_ontology.enabled must be boolean")
        for key in ("max_features_per_role", "max_alternatives_per_feature",
                    "nuisance_crossfit_folds", "minimum_training_rows",
                    "minimum_overlap_rows", "max_prompt_chars"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"estimand_ontology.{key} must be a positive integer")
        if self.nuisance_crossfit_folds < 2:
            raise ValueError("estimand_ontology.nuisance_crossfit_folds must be at least two")
        for key in ("minimum_nonmissing_fraction", "minimum_relative_gain",
                    "minimum_winning_fold_fraction", "maximum_propensity_loss_increase",
                    "maximum_overlap_fraction_drop", "paired_se_multiplier", "ridge_penalty"):
            value = getattr(self, key)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f"estimand_ontology.{key} must be finite and nonnegative")
            if key not in {"paired_se_multiplier", "ridge_penalty"} and value > 1:
                raise ValueError(f"estimand_ontology.{key} must be at most one")
        if self.ridge_penalty <= 0:
            raise ValueError("estimand_ontology.ridge_penalty must be positive")

    def public_dict(self):
        return asdict(self)


def estimand_ontology_config_from_mapping(raw=None):
    if raw is not None and not isinstance(raw, Mapping):
        raise ValueError("stage2.estimand_ontology must be an object")
    raw = dict(raw or {})
    unknown = set(raw) - set(EstimandOntologyConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown estimand_ontology settings: {sorted(unknown)}")
    result = EstimandOntologyConfig(**raw)
    result.validate()
    return result
