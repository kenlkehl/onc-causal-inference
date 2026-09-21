"""Lightweight scientific policy for the opt-in Stage 2 evidence ensemble."""

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

SCHEMA_VERSION = "stage2_multi_model_selection_v1"
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

    def validate(self) -> None:
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
        return {"schema_version": SCHEMA_VERSION, **asdict(self)}


def multi_model_config_from_mapping(value: Mapping[str, Any] | None) -> Stage2MultiModelConfig:
    if value is not None and not isinstance(value, Mapping):
        raise ValueError("stage2.statistical_selection.multi_model must be an object")
    raw = dict(value or {})
    if raw.pop("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise ValueError("unsupported multi_model schema_version")
    unknown = set(raw) - set(Stage2MultiModelConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unsupported multi_model fields: {sorted(unknown)}")
    if "l1_ratios" in raw:
        if not isinstance(raw["l1_ratios"], (list, tuple)):
            raise ValueError("multi_model.l1_ratios must be a list")
        raw["l1_ratios"] = tuple(raw["l1_ratios"])
    result = Stage2MultiModelConfig(**raw)
    result.validate()
    return result
