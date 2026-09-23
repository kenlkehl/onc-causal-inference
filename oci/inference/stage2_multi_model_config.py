"""Lightweight scientific policy for the opt-in Stage 2 evidence ensemble."""

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping

SCHEMA_VERSION = "stage2_multi_model_selection_v3"
PROMPT_VERSION = "stage2_multi_model_themes_v1"
FAMILIES = (
    "univariable",
    "penalized_main",
    "penalized_interactions",
    "orthogonal_linear",
    "univariable_rlearner",
    "predictive_forest",
    "causal_forest",
)
FINAL_ESTIMATORS = ("causal_forest", "linear_interactions")


@dataclass(frozen=True)
class ModifierCountConfig:
    """Nested R-loss tuning of additional, nonlocked modifier candidates."""

    enabled: bool = True
    candidate_counts: tuple[int, ...] = (0, 4, 8, 12, 16, 24, 32)
    max_ranked_modifiers: int = 64
    selection_rule: str = "minimum_r_loss"
    forest_seeds: int = 3
    estimators: tuple[str, ...] = FINAL_ESTIMATORS

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("multi_model.modifier_count.enabled must be boolean")
        if (
            not isinstance(self.estimators, tuple)
            or not self.estimators
            or any(not isinstance(v, str) or v not in FINAL_ESTIMATORS for v in self.estimators)
            or len(set(self.estimators)) != len(self.estimators)
        ):
            raise ValueError(
                "modifier_count.estimators must be a nonempty unique list of causal_forest and/or linear_interactions"
            )
        for name in ("max_ranked_modifiers", "forest_seeds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"modifier_count.{name} must be a positive integer")
        if (
            not isinstance(self.candidate_counts, tuple)
            or not self.candidate_counts
            or any(
                isinstance(k, bool) or not isinstance(k, int) or k < 0
                for k in self.candidate_counts
            )
            or tuple(sorted(set(self.candidate_counts))) != self.candidate_counts
            or self.candidate_counts[0] != 0
            or self.candidate_counts[-1] > self.max_ranked_modifiers
        ):
            raise ValueError(
                "modifier_count.candidate_counts must be sorted unique integers starting at 0 and at most max_ranked_modifiers"
            )
        if not isinstance(self.selection_rule, str) or self.selection_rule not in {
            "minimum_r_loss",
            "one_standard_error",
        }:
            raise ValueError(
                "modifier_count.selection_rule must be minimum_r_loss or one_standard_error"
            )

    def public_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "candidate_counts": list(self.candidate_counts),
            "estimators": list(self.estimators),
        }


@dataclass(frozen=True)
class Stage2MultiModelConfig:
    repeats: int = 3
    row_fraction: float = 0.8
    l1_ratios: tuple[float, ...] = (0.2, 0.8, 1.0)
    regularization_grid_size: int = 8
    forest_trees: int = 200
    forest_min_samples_leaf: int = 10
    feature_subset_size: int = 32
    permutation_repeats: int = 3
    nominal_p_threshold: float = 0.05
    q_threshold: float = 0.1
    max_prompt_chars: int = 100_000
    modifier_count: ModifierCountConfig = field(default_factory=ModifierCountConfig)

    def validate(self) -> None:
        if not isinstance(self.modifier_count, ModifierCountConfig):
            raise ValueError("multi_model.modifier_count must be a ModifierCountConfig object")
        self.modifier_count.validate()
        for name in (
            "repeats",
            "regularization_grid_size",
            "forest_trees",
            "forest_min_samples_leaf",
            "feature_subset_size",
            "permutation_repeats",
            "max_prompt_chars",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(
                    f"stage2.statistical_selection.multi_model.{name} must be positive integer"
                )
        if self.regularization_grid_size < 3 or self.forest_trees < 4:
            raise ValueError("multi_model requires at least 3 regularization values and 4 trees")
        if self.max_prompt_chars < 4_000:
            raise ValueError("multi_model.max_prompt_chars must be at least 4000")
        for name in ("row_fraction", "nominal_p_threshold", "q_threshold"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 < value <= 1
            ):
                raise ValueError(f"multi_model.{name} must be in (0, 1]")
        if not self.l1_ratios or any(
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            or not 0 < v <= 1
            for v in self.l1_ratios
        ):
            raise ValueError("multi_model.l1_ratios must contain finite values in (0, 1]")

    def public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            **asdict(self),
            "modifier_count": self.modifier_count.public_dict(),
        }


def multi_model_config_from_mapping(
    value: Mapping[str, Any] | None,
) -> Stage2MultiModelConfig:
    if value is not None and not isinstance(value, Mapping):
        raise ValueError("stage2.statistical_selection.multi_model must be an object")
    raw = dict(value or {})
    if raw.pop("schema_version", SCHEMA_VERSION) not in {
        SCHEMA_VERSION,
        "stage2_multi_model_selection_v1",
        "stage2_multi_model_selection_v2",
    }:
        raise ValueError("unsupported multi_model schema_version")
    unknown = set(raw) - set(Stage2MultiModelConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unsupported multi_model fields: {sorted(unknown)}")
    if "l1_ratios" in raw:
        if not isinstance(raw["l1_ratios"], (list, tuple)):
            raise ValueError("multi_model.l1_ratios must be a list")
        raw["l1_ratios"] = tuple(raw["l1_ratios"])
    if "modifier_count" in raw:
        count = raw["modifier_count"]
        if not isinstance(count, Mapping):
            raise ValueError("multi_model.modifier_count must be an object")
        count = dict(count)
        unknown = set(count) - set(ModifierCountConfig.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unsupported modifier_count fields: {sorted(unknown)}")
        for name in ("candidate_counts", "estimators"):
            if name in count:
                if not isinstance(count[name], (list, tuple)):
                    raise ValueError(f"modifier_count.{name} must be a list")
                count[name] = tuple(count[name])
        raw["modifier_count"] = ModifierCountConfig(**count)
    result = Stage2MultiModelConfig(**raw)
    result.validate()
    return result
